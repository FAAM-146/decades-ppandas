import datetime
import os
import requests
import glob
import subprocess
import shutil
import tempfile

import pandas as pd
import fpdf

from pylatex.utils import escape_latex as _el
from PIL import Image

from .overview import make_plot

META_URL = 'https://www.faam.ac.uk/gluxe/flights/{fltnum}/meta?token={token}'
CHAT_URL = 'https://www.faam.ac.uk/gluxe/chat/get?since={since}&until={until}'
BRIEF_URL = 'https://www.faam.ac.uk/gluxe/flights/{fltnum}/sortiebrief'
SUMMARY_URL = 'https://www.faam.ac.uk/gluxe/fltsum/get?fltnum={fltnum}'

class ReportCompiler(object):

    def __init__(self, dataset, flight_number=None, qa_dir=None,
                 output_dir=None, token=None, flight_folder=None):

        self.dataset = dataset
        self.flight_number = flight_number
        self.flight_folder = flight_folder
        self.qa_dir = qa_dir or os.getcwd()
        self.token = token
        self.output_dir = output_dir or os.getcwd()

        self._get_meta()
        self._get_chat()
        self._get_summary()

    def _get_summary(self):
        url = SUMMARY_URL.format(fltnum=self.flight_number)
        response = requests.get(url)
        summary = response.json()
        self.summary = summary[::-1]

    def _get_meta(self):
        url = META_URL.format(fltnum=self.flight_number, token=self.token)
        response = requests.get(url)
        meta = response.json()

        self._meta = meta

        self.crewlist = meta['crew']
        self.timings = meta['timings']
        self.project = meta['project']

        try:
            self.takeoff_time = datetime.datetime.strptime(
                meta['takeoff']['time_utc'],
                '%Y-%m-%dT%H:%M:%S'
            )
        except Exception:
            self.takeoff_time = self.dataset.date

        try:
            self.landing_time = datetime.datetime.strptime(
                meta['landing']['time_utc'],
                '%Y-%m-%dT%H:%M:%S'
            )
        except Exception:
            self.landing_time = self.dataset.date

        try:
            self.date = self.takeoff_time.date()
        except Exception:
            self.date = self.dataset.date


    def _get_chat(self):
        _buffer = datetime.timedelta(minutes=30)
        _start = (self.takeoff_time - _buffer).strftime('%Y-%m-%dT%H:%M:%S')
        _end = (self.landing_time + _buffer).strftime('%Y-%m-%dT%H:%M:%S')

        url = CHAT_URL.format(since=_start, until=_end)
        response = requests.get(url)
        _chat = response.json()

        _messages = []
        for message in _chat:
            _fields = message['fields']
            if _fields['username']:
                _messages.append((
                    datetime.datetime.strptime(
                        _fields['time'],
                        '%Y-%m-%dT%H:%M:%S.%f'
                    ),
                    _fields['username'],
                    _fields['message']
                ))

        self.chatlog = sorted(_messages, key=lambda x: x[0])


    def _head(self):
        return r"""
\documentclass{article}
\usepackage[x11names]{xcolor}
\usepackage{pict2e}% to allow any radius
\usepackage{fontspec}
\usepackage{longtable}
\usepackage{rotating}
\usepackage{pdfpages}
\usepackage{pdflscape}
\usepackage[a4paper,left=20mm,right=20mm,bottom=30mm]{geometry}
%\setmainfont[Path=/usr/local/share/fonts/n/]{Nexa_Light}
\newfontfamily\nexalight[Path=/usr/local/share/fonts/n/]{Nexa_Light}
\newfontfamily\nexabold[Path=/usr/local/share/fonts/n/]{Nexa_Bold}
\setmainfont{nexalight}

\definecolor{FAAMDarkBlue}{HTML}{252243}
\definecolor{FAAMLightBlue}{HTML}{0ABBEF}

\usepackage{eso-pic}
\newcommand\BackgroundPic{%
    \put(305,-260){
        \color{FAAMLightBlue}\circle*{900}
    }
    \put(305,-260){
        \color{FAAMDarkBlue}\circle*{760}
    }
}
\begin{document}
\pagenumbering{gobble}
\AddToShipoutPicture*{\BackgroundPic}
\color{FAAMDarkBlue}
"""

    def _titlepage(self):
        return r"""
\begin{titlepage}
    %    \pagecolor{FAAMDarkBlue}
    \begin{center}
        {\nexabold {\Huge{F A A M}}}\\
        \vspace{2ex}
        {A I R B O R N E}\\
        {L A B O R A T O R Y}

        \vspace{12ex}
        {\Huge F L I G H T}\\
        \vspace{1ex}
        {\Huge R E P O R T}\\
        \vspace{8ex}
        {\Large {TAG_FLIGHTNUM}}\\
        \vspace{2ex}
        {\large {TAG_PROJECT}}\\
        \vspace{8ex}
        {TAG_DATE}
    \end{center}
\end{titlepage}
""".replace(
    'TAG_FLIGHTNUM', self.flight_number.upper()
).replace(
    'TAG_PROJECT', self.project
).replace(
    'TAG_DATE', self.date.strftime('%A, %B %-d, %Y')
)

    def _sortiebrief(self):
        url = BRIEF_URL.format(fltnum=self.flight_number)
        response = requests.get(url)
        with open('brief.pdf', 'wb') as pdf:
            pdf.write(response.content)

        if not os.stat('brief.pdf').st_size:
            return ''

        _retstr = r'\includepdf[pages=-]{brief}'
        return _retstr

    def _chatlog(self):
        _retstr = ''
        _retstr += r'\section{Chat Log}' + '\n'
        _retstr += r'\begin{longtable}{lll}' + '\n'
        for message in self.chatlog:
            _retstr += r'{time} & \textbf{{ {user} }} & '.format(
                time=message[0].strftime('%H%M.%S'),
                user=_el(message[1]),
            )
            _retstr += r'\begin{minipage}{100mm}'
            _retstr += _el(message[2])
            _retstr += r'\end{minipage}\\[.3cm]' + '\n'
        _retstr += r'\end{longtable}' + '\n'
        return _retstr

    def _timings(self):
        _retstr = ''
        _retstr += r'\section{Timings}' + '\n'
        _retstr += r'\begin{longtable}{ll}' + '\n'
        for timing in self.timings:
            _retstr +=  r'\textbf{{ {time} }} & {event}'.format(
                time=_el(datetime.datetime.strptime(
                    timing['time'], '%Y-%m-%dT%H:%M:%S'
                ).strftime('%H%M Z')),
                event=_el(timing['event'])
            ) + r'\\' + '\n'
        _retstr += r'\end{longtable}'
        return _retstr

    def _flight_summary(self):
        _retstr = r'\begin{landscape}'
        _retstr += r'\section{Flight Summary}' + '\n'
        _retstr += r'{\scriptsize' + '\n'
        _retstr += r'\begin{longtable}{llllllllllll}' + '\n'

        _retstr += r'\hline' + '\n'
        _retstr += (r'Event & Start & Start lat & Start lon & Start alt & '
                    r'Start hdg & End & End lat & End lon & End alt & '
                    r'End hdg & Comment \\') + '\n'
        _retstr += r'\hline' + '\n'

        for evt in self.summary:
            _evt = evt['fields']

            for key, val in _evt.items():
                if val is None:
                    _evt[key] = ''

            _str = r'{evt} & {stime} & {slat:0.2f} & {slon:0.2f} & {salt} & {shdg} & '
            if _evt['stop_time']:
                _str += r'{etime} & {elat:0.2f} & {elon:0.2f} & {ealt} & {ehdg} & '
            else:
                _str += r'{etime} & {elat} & {elon} & {ealt} & {ehdg} &'
            _str += r'{comment}\\' + '\n'

            _retstr += _str.format(
                evt=_el(_evt['event']),
                stime=_el(_evt['start_time'][-8:]),
                slat=_evt['start_lat'],
                slon=_evt['start_lon'],
                salt=_evt['start_alt'],
                shdg=_evt['start_heading'],
                etime=_el(_evt['stop_time'][-8:]),
                elat=_evt['stop_lat'],
                elon=_evt['stop_lon'],
                ealt=_evt['stop_alt'],
                ehdg=_evt['stop_heading'],
                comment=_el(_evt['comment'])
            )
        _retstr += '\n'
        _retstr += r'\end{longtable}' + '\n'
        _retstr += r'}'
        _retstr += r'\end{landscape}'
        return _retstr

    def _qaqc(self):
        _retstr = ''
        _retstr += r'\begin{center}'
        for _file in glob.glob(os.path.join(self.qa_dir, '*.pdf')):
            _retstr += r'\includepdf[pages=-]{' + _file.replace('.pdf', '') + r'}' + '\n'
        _retstr += r'\end{center}' + '\n'
        return _retstr

    def _appendix(self):
        if not self.flight_folder:
            return ''

        cwd = os.getcwd()
        outputs = []
        for _root, _dirs, _files in os.walk(self.flight_folder):
            for _file in _files:
                if ' ' in _file:
                    shutil.move(
                        os.path.join(_root, _file),
                        os.path.join(_root, _file.replace(' ', '_'))
                    )
                    _file = _file.replace(' ', '_')

                _filepath = os.path.join(_root, _file)
                if _file[-3:].lower() in ['png', 'jpg']:
                    outputs.append(_filepath)

        _retstr = ''
        for _op in outputs:
            _retstr += r'\begin{center}' + '\n'
            _retstr += r'\begin{figure}' + '\n'
            _retstr += r'\includegraphics[width=.8\pagewidth]{' + _op + '}\n'
            _retstr += r'\caption{' + _el(os.path.basename(_op)) + '}\n'
            _retstr += r'\end{figure}' + '\n'
            _retstr += r'\end{center}' + '\n'

        return _retstr

    def _overview(self):
        lat = self.dataset['LAT_GIN'].data.asfreq('1S')
        lon = self.dataset['LON_GIN'].data.asfreq('1S')
        alt = self.dataset['ALT_GIN'].data.asfreq('1S')
        flight_data = pd.DataFrame({
            'alt': alt, 'lat': lat, 'lon': lon
        }, index=alt.index)
        try:
            make_plot(flight_data, outfile='overview.pdf', meta=self._meta)
        except Exception:
            return ''
        _retstr = r'\begin{center}'
        _retstr += r'\includepdf[pages=-]{overview.pdf}'
        _retstr += r'\end{center}'
        return _retstr

    def _crew(self):
        _retstr = ''
        _retstr += r'\section{Crew}' + '\n'
        _retstr += r'\begin{longtable}{ll}' + '\n'

        for member in self.crewlist:
            _retstr += '{role} & {name} ({institute})'.format(
                role=_el(member['role']),
                name=_el(member['name']),
                institute=_el(member['institute'])
            )
            _retstr += r'\\' + '\n'

        _retstr += r'\end{longtable}' + '\n'

        return _retstr

    def _tail(self):
        return r'\end{document}'

    def build(self, filename):
        if not filename.endswith('.tex'):
            raise ValueError('filename must be a .tex document')
        self.filename = filename
        with open(filename, 'w') as f:
            f.write(self._head())
            f.write(self._titlepage())
            f.write(self._sortiebrief())
            f.write(self._overview())
            f.write(self._crew())
            f.write(self._timings())
            f.write(self._flight_summary())
            f.write(self._chatlog())
            f.write(self._qaqc())
            f.write(self._appendix())
            f.write(self._tail())

    def compile(self):
        cmd = ['lualatex', '-interaction', 'nonstopmode', self.filename]
        with open(os.devnull, 'w') as devnull:
            for i in range(2):
                subprocess.call(cmd, stdout=devnull, stderr=devnull)

        shutil.move(
            self.filename.replace('.tex', '.pdf'),
            os.path.join(
                self.output_dir,
                'flight-report_faam_{date}_r0_{fltnum}.pdf'.format(
                    date=self.date.strftime('%Y%m%d'),
                    fltnum=self.flight_number.lower()
                )
            )
        )

    def make(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as _tmp:
            try:
                os.chdir(_tmp)
                self.build('temp.tex')
                self.compile()
            except Exception:
                raise
            finally:
                os.chdir(cwd)

