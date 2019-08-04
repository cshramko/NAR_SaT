#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NAR S&T motor test data processor"""

__author__ = 'Christopher D. Shramko'
__copyright__ = 'Copyright 2019, Christopher D. Shramko and National Association of Rocketry'
__organization__ = 'National Association of Rocketry, Standards and Testing Committee'
__contact__ = 'chris@shramko.net'
__credits__ = ['Christopher D. Shramko', 'Guy Wadsworth', 'Edward Pattison-Gordon', 'John (Jack) J Kane', 'Ken Blade']
__license__ = 'GNU General Public License 3.0 (GPLv3)'
__version__ = '0.1'
__maintainer__ = 'Christopher D. Shramko'
__email__ = 'chris@shramko.net'
__status__ = 'Development'

# TODO: process all files in a directory, auto-grouping by engine type
# TODO: multi-run averaging
# TODO: publishing code to GitHub as a publicly-readable repository, if permitted
# TODO: fix logging disappearing to bitbucket and convert remaining print statements

# TODO: standard deviations - see code for guidelines
# TODO: ejection charge delay verification within deviation allowances
# TODO: is thrust deviation within limits
# TODO: is burn time deviation within limits
# TODO: is signal/noise ratio within limits
# TODO: is propellant mass deviation limits
# TODO: reduce to best fit 32 points maintaining total impulse

# TODO: tag version 0.5 when we run in tandem with legacy but still use legacy results
# TODO: tag version 0.9 when we run in tandem with legacy and use our results
# TODO: tag version 1.0 when we have completed migration and no longer run legacy

# TODO: assumption of initial cert or re-cert based on # of files provided
# TODO: minimum run count verification for retest and initial cert and each delay
# TODO: auto-smoothing taking into account noise level calculation & maintaining total impulse
# TODO: do we need to smooth before we reject noise and max thrust
# TODO: chuffing detection and discard all but final burn
# TODO: q-jet igniters have caused spurious max thrust, how do we handle for this
# TODO: spurious data-point rejection (off curve & outside noise level) (only if it affects #'s)

# TODO: import DAT header map from DAT-Format specification file
# TODO: single PDF report / conclusion (per engine type)
# TODO: import data directly from test stand to program using python drivers from OEM
# TODO: formatted report (markup and/or YAML and/or JSON)

#
# built-in modules
#

import os
import sys
import argparse
import textwrap
import datetime
import re
import logging
import logging.handlers
# import itertools

#
# third-party modules
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import zipfile

#
# project modules
#

#
# constants (globals) & globals
#

IGNORE_BELOW = 0.05  # ignore data-points less that this * max impulse
IGNITION_BACKOFF = 0.10  # ignore last initial skipped points * this for noise calculations
RE_NUMERIC = re.compile(r'^\s*[\-]?\d*[,.]?\d*\s*$')  # is string numeric and not scientific
debug = False
logger = logging.getLogger()


#
# project classes
#


class MotorTest:
    """Contains the measured and calculated data for a motor test."""

    def __init__(self):
        """Create a new instance of this object with result status of UNKNOWN."""

        self.comments = ''
        self.result = 'UNKNOWN'
        self.file_name = ''
        self.operator = ''
        self.mfg = ''
        self.test_time = ''
        self.test_date = ''
        self.site_elevation = -1
        self.motor_type = ''
        self.casing_code = ''
        self.propellant_type = ''
        self.propellant_mass = -1.0
        self.casing_diameter = -1
        self.casing_length = -1
        self.initial_mass = -1.0
        self.burned_out_mass = -1.0
        self.test_temperature = -1.0
        self.max_casing_temperature = -1.0
        self.number_data_points = -1
        self.ejection_delay = -1
        self.max_liftoff_weight = ''
        self.reserved_7 = ''
        self.reserved_6 = ''
        self.reserved_5 = ''
        self.reserved_4 = ''
        self.reserved_3 = ''
        self.reserved_2 = ''
        self.reserved_1 = ''
        self.graph_points_per_sec = -1
        self.max_test_length = -1
        self.data_point_averaging = -1
        self.scan_rate_per_sec = -1

    def set_result(self, new_status='UNKNOWN'):
        """Set result to more dire of current or new status, with
        possible values (from least to most dire) of OK UNKNOWN WARN FAIL.
        Ignore any invalid new status.
        """

        possible_values = ['OK', 'UNKNOWN', 'WARN', 'FAIL']
        current_index = possible_values.index(self.result)
        try:
            new_index = possible_values.index(new_status)
            if new_index > current_index:
                self.result = new_status
        except ValueError:
            log.warning('Invalid MotorTest.result status: ' + str(new_status))


class SessionWorkSpace:
    """Maintains state and work queue for a single session"""

    def __init__(self, task, target):
        self.task = task
        self.target = target
        self.motorlist = []


#
# project functions
#

#
# logging functions
#

def _logging_setup(task_name, target_type, target_name,
                   logfile_silent, console_silent, debug_mode):
    """ Set up dual logging to console and to logfile.
    Set log file logging name to YYYYMMDD-HHMM-[TASK_NAME]-[TARGET_NAME].log.
    Set console logging level to DEBUG.
    Set stdout logging level to DEBUG if if debug_mode, else INFO.
    """

    # set base log level (not theoretically necessary)
    log.setLevel(logging.DEBUG)

    # construct the logfile name

    target_name = target_name[:-4]  # remove extension
    if target_type == 'session':
        target_name = 'session'
    t = datetime.datetime.now()
    logfile = '{year:04d}{mon:02d}{day:02d}-' \
              '{hour:02d}{min:02d}-{task}-{target}.log'.format(
                    year=t.year, mon=t.month, day=t.day,
                    hour=t.hour, min=t.minute, sec=t.second,
                    task=task_name, target=target_name)

    # set up logging to file

    if not logfile_silent:
        filehandler = logging.handlers.RotatingFileHandler(
            filename=logfile,
            maxBytes=10 * 1024 * 1024,
            backupCount=100)
        fileformatter = logging.Formatter(
            '%(asctime)s %(module)s %(funcName)s %(levelname)s %(message)s',
            '%Y-%m-%d %H:%M:%S')
        filehandler.setFormatter(fileformatter)
        filehandler.setLevel('DEBUG')
        log.addHandler(filehandler)

    # set up logging to console

    if not console_silent:
        streamhandler = logging.StreamHandler()
        streamformatter = logging.Formatter('%(message)s')
        streamhandler.setFormatter(streamformatter)
        # if debug_mode:
        #     streamhandler.setLevel('DEBUG')
        # else:
        #     streamhandler.setLevel('INFO')
        streamhandler.setLevel('INFO')
        log.addHandler(streamhandler)

    # log logging status

    if logfile_silent:
        log.info('Logging to file is silenced')
    else:
        log.info('Logging to ' + logfile + '.')

    if console_silent:
        log.info('Console messaging is silenced')
    else:
        log.info('Logging to console.')

#
# check functions
#

def check_motor_file(motor_file_name):
    """Verify existence, readability, and sufficient mimimnum row count.
    Return success/failure as Boolean and reason message.
    """

    # verify existence and readability of motor_file_name

    if not (os.path.isfile(motor_file_name) and
            os.access(motor_file_name, os.R_OK)):
        message = 'Motor data file not readable: ' + motor_file_name
        log.error(message)
        return False, message

    # read data from motor_file_name, ensuring sufficient minimum rows

    with open(motor_file_name) as motor_file:
        md = motor_file.readlines()
    if len(md) < 64:
        message = 'Insufficient Header Row Count in ' + motor_file_name
        log.error(message)
        return False, message

    return True, ''

#
# motor functions
#

def process_motor(motor_file_name):
    """Process a motor file and create summary and graph files.
    Return results as a MotorTest.
    """

    # Initiate Motor Test Object

    mt = MotorTest()

    # check for valid-looking motor file

    success, message = check_motor_file(motor_file_name)
    if not success:
        mt.set_result('FAIL')
        mt.comments = mt.comments + ' ' + message
        return mt

    # read data from motor_file_name, stripping leading and trailing whitespace

    with open(motor_file_name) as motor_file:
        md = motor_file.readlines()
    md = [x.strip() for x in md]

    # parse motor file header rows

    header_order = [('file_name', 's'),
                    ('operator', 's'),
                    ('mfg', 's'),
                    ('test_time', 's'),
                    ('test_date', 's'),
                    ('site_elevation', 'i'),
                    ('motor_type', 's'),
                    ('casing_code', 's'),
                    ('propellant_type', 's'),
                    ('propellant_mass', 'f'),
                    ('casing_diameter', 'i'),
                    ('casing_length', 'i'),
                    ('initial_mass', 'f'),
                    ('burned_out_mass', 'f'),
                    ('test_temperature', 'f'),
                    ('max_casing_temperature', 'f'),
                    ('number_data_points', 'i'),
                    ('ejection_delay', 'i'),
                    ('max_liftoff_weight', 's'),
                    ('reserved_7', 's'),
                    ('reserved_6', 's'),
                    ('reserved_5', 's'),
                    ('reserved_4', 's'),
                    ('reserved_3', 's'),
                    ('reserved_2', 's'),
                    ('reserved_1', 's'),
                    ('graph_points_per_sec', 'i'),
                    ('max_test_length', 'i'),
                    ('data_point_averaging', 'i'),
                    ('scan_rate_per_sec', 'i')
                    ]

    for row, attr in enumerate(header_order):

        value = md[row]
        (attr_name, attr_type) = attr

        if value and not value == '*':
            if attr_type == 's':
                setattr(mt, attr_name, value)
            if attr_type == 'f':
                setattr(mt, attr_name, float(value))
            if attr_type == 'i':
                setattr(mt, attr_name, int(value))

        logging.debug(attr_name + ' : ' + value)

    # parse motor file data rows

    mt.graph_points = md[30:]
    if not hasattr(mt, 'graph_points'):
        mt.graph_points = [0]
    mt.found_graph_points = len(mt.graph_points)
    log.debug('found graph points: ' + str(mt.found_graph_points))

    # motor file data sanity tests

    if mt.found_graph_points < 32:
        message = 'Too few Data Points. '
        mt.comments = mt.comments + message
        mt.set_result('FAIL')
        log.error(message)
    if mt.found_graph_points != mt.number_data_points:
        message = 'Found Graph Points != Number Data Points. '
        mt.comments = mt.comments + message
        mt.set_result('WARN')
        log.debug(message)
    if (mt.scan_rate_per_sec / mt.data_point_averaging) != mt.graph_points_per_sec:
        message = 'Data rate mismatch (scan / averaging != graph). '
        mt.comments = mt.comments + message
        mt.set_result('WARN')
        log.debug(message)
    if (mt.graph_points_per_sec * mt.max_test_length) < mt.number_data_points:
        message = 'Found Graph Points exceed maximum test length. '
        mt.comments = mt.comments + message
        mt.set_result('WARN')
        log.debug(message)

    # convert data points to floats and find data max

    mt.graph_points = [float(x) for x in mt.graph_points]
    mt.max_impulse = round(float(max(mt.graph_points)), 2)
    log.debug('Max Impulse: ' + str(mt.max_impulse))

    # calculate min impulse absolute (and percent)

    mt.min_tracked_impulse = mt.max_impulse * IGNORE_BELOW
    mt.min_tracked_impulse_percent = mt.min_tracked_impulse * 100
    log.debug('Min Tracked Impulse: ' + str(mt.min_tracked_impulse_percent) + ' %')
    log.debug('Min Tracked Impulse: ' + str(mt.min_tracked_impulse))

    # remove leading unneeded data-points
    # calculate noise, backing off from ignition area to avoid rough ignition noise

    i = 0
    for i, impulse in enumerate(mt.graph_points):
        if impulse >= mt.min_tracked_impulse:
            break
    mt.leading_skipped = i
    mt.leading_points = mt.graph_points[:int((1 - IGNITION_BACKOFF) * i)]
    mt.graph_points = mt.graph_points[i:]
    mt.noise_level = round((max(mt.leading_points) - min(mt.leading_points)) / 2, 3)

    # calculate baseline shift and shift data-point baseline

    mt.baseline_shift = np.mean(mt.leading_points)
    mt.graph_points = [(x - mt.baseline_shift) for x in mt.graph_points]
    log.debug('Leading Data Points Skipped: ' + str(mt.leading_skipped))
    log.debug('Noise Level: +/- ' + str(mt.noise_level))
    log.debug('Max Leading Point: ' + str(max(mt.leading_points)))
    log.debug('Min Leading Point: ' + str(min(mt.leading_points)))
    log.debug('Baseline Shift: ' + str(mt.baseline_shift))

    # remove trailing unneeded data-points

    i = 0
    for i, impulse in enumerate(mt.graph_points):
        if impulse < mt.min_tracked_impulse:
            break
    mt.trailing_skipped = len(mt.graph_points) - i + 1
    mt.trailing_points = mt.graph_points[i:]
    mt.graph_points = mt.graph_points[:(i - 1)]
    mt.graph_points.insert(0, 0.0)
    mt.graph_points.append(0.0)
    mt.points_kept = len(mt.graph_points) - 1
    log.debug('Trailing Data Points Dkipped: ' + str(mt.trailing_skipped))
    log.debug('Data Points Remaining: ' + str(len(mt.graph_points)))

    # calculate timestamps

    mt.time_interval = round((1.0 / mt.graph_points_per_sec), 3)
    mt.graph_times = [(x * mt.time_interval) for x in range(0, mt.points_kept + 1)]
    log.debug('Time Interval: ' + str(mt.time_interval))

    # other calculations and console output

    print('------------------------------')
    print('Processed File:   ' + motor_file_name)

    mt.total_impulse = round(simps(mt.graph_points, dx=mt.time_interval), 3)
    print('Total Impulse:    ' + str(mt.total_impulse))

    print('Peak Thrust:      ' + str(mt.max_impulse))

    mt.burn_time = round(mt.graph_times[len(mt.graph_times) - 1], 2)
    print('Burn Time:        ' + str(mt.burn_time))

    eject_indicator_level = 5 * mt.min_tracked_impulse
    if abs(max(mt.trailing_points)) < eject_indicator_level:
        mt.calculated_ejection_delay = 0
    else:
        for i, impulse in enumerate(mt.trailing_points):
            if abs(impulse) > eject_indicator_level:
                break
        # mt.calculated_ejection_delay = mt.burn_time + (i * mt.time_interval)
        mt.calculated_ejection_delay = i * mt.time_interval
        mt.calculated_ejection_delay = round(mt.calculated_ejection_delay, 2)
    print('Ejection Delay:   ' + str(mt.calculated_ejection_delay))

    mt.average_impulse = round(float(np.mean(mt.graph_points)), 2)
    print('Average Thrust:   ' + str(mt.average_impulse))

    if mt.result == 'UNKNOWN':
        mt.result = 'OK'
    print('Result:           ' + mt.result)

    if mt.comments == '':
        mt.comments = 'None'
    print('Comments:         ' + mt.comments)

    # plot a graph

    plt.title(mt.motor_type)
    plt.xlabel('Time (sec)')
    plt.ylabel('Thrust (N)')
    plt.ylim(top=100)
    plt.yticks(np.arange(0, 100, step=25))
    plt.minorticks_on()
    plt.grid(True, which='major', axis='both')
    plt.plot(mt.graph_times, mt.graph_points)

    # write graph to file
    plt.savefig(mt.file_name + '.png')
    print('Thrustcurve:      ' + mt.file_name + '.png')

    # write data to file

    with open(mt.file_name + '.txt', 'w') as f:
        f.write('\nMotor Type:          ' + mt.motor_type)
        f.write('\nManufacturer:        ' + mt.mfg)
        f.write('\nDelays:              ' + str(mt.ejection_delay))
        f.write('\n')
        f.write('\nPropellant Type:     ' + mt.propellant_type)
        f.write('\nPropellant Mass:     ' + str(mt.propellant_mass))
        f.write('\nMass After Firing:   ' + str(mt.burned_out_mass))
        f.write('\n')
        f.write('\nCasing Diameter:     ' + str(mt.casing_diameter))
        f.write('\nCasing Length:       ' + str(mt.casing_length))
        f.write('\nCasing Code:         ' + mt.casing_code)
        f.write('\n')
        f.write('\nFile Name:           ' + mt.file_name)
        f.write('\nDate Tested:         ' + mt.test_date + ' ' + mt.test_time)
        f.write('\n')
        f.write('\nTotal Impulse:       ' + str(mt.total_impulse))
        f.write('\nPeak Thrust:         ' + str(mt.max_impulse))
        f.write('\nBurn Time:           ' + str(mt.burn_time))
        f.write('\nAverage Thrust:      ' + str(mt.average_impulse))
        f.write('\nEjection Delay:      ' + str(mt.calculated_ejection_delay))
        f.write('\n')
        f.write('\nData Noise (+/-):    ' + str(mt.noise_level))
        f.write('\n')
        f.write('\nProcessing Date:     ' + str(datetime.datetime.now())[:19])
        f.write('\nCertification Type:  ' + 'Model Rocket')
        f.write('\nProcessing Result:   ' + mt.result)
        f.write('\n')
        f.write('\nComments:            ' + mt.comments)
        f.write('\n')
        f.write('\nNote: -1 indicates no value provided or available.')
        f.write('\n')
        f.write('\n')
    print('Report:           ' + mt.file_name + '.txt')

    return mt


def report_motor(motor_file_name):
    """Build a PDF report from a motor file's processed results."""

    return True


def bundle_motor(motor_file_name):
    """Bundle all files for a processed (and reported) motor into a ZIP."""

    # check for valid-looking motor file

    success, message = check_motor_file(motor_file_name)
    if not success:
        return False

    motor_file_name = motor_file_name[:-4]

    log.info('Bundling files related to: ' + motor_file_name)

    zip_archive = motor_file_name + '.zip'
    zip_pattern = motor_file_name + '.'
    zip_targets = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk('.'):
        for file in f:
            if zip_pattern in file:
                if not file == zip_archive:
                    zip_targets.append(file)
                    log.info('Found target: ' + file)
    if not zip_targets:
        log.error('No targets found to bundle ' + motor_file_name)
        return False

    with zipfile.ZipFile(zip_archive, mode='w') as bundle_zip:
        for target_name in zip_targets:
            bundle_zip.write(target_name)
            print('Bundled: ' + target_name)

    print('Bundle: ' + zip_archive)

    return True

#
# session functions
#

def process_session(target):
    """Process all motor files in a directory as a testing session.
    Create a Session folder with motor type folders, session log, and summary.
    Create motor summaries and graphs within the motor type folders.
    """

    return True


def report_session(target):
    """Build PDF reports from a session's motor type results.
    Create a Session summary PDF.
    """

    return True


def bundle_session(target):
    """Bundle all files for a processed (and reported) session into a ZIP."""

    return True

#
# main function
#

def main():
    """motortest main function

    Parse parameters and perform the appropriate task:
        process     verify and analyze motor data fils),
                    creating summary report(s) and graph(s)
        report      create a PDF report of the results
        bundle      bundle all related files into a ZIP archive

    on the appropriate object:
        motor       a single specified motor data file
        session     all motor data files within the specified directory
    """

    # setup argument parser

    main_parser = argparse.ArgumentParser(
        description='Process rocket motor test data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        task, target-type, and target must be specified.

        Example:
           %(prog)s process motor test1.dat
           %(prog)s report session .\\2015\\feb --show=warning
           %(prog)s bundle session . --log=DEBUG --silent
        '''))

    main_parser.add_argument('task', type=str.lower,
                             choices=['process', 'report', 'bundle'],
                             help='which task to perform')

    main_parser.add_argument('target_type', type=str.lower,
                             choices=['motor', 'session'],
                             help='which type to target')

    main_parser.add_argument('target',
                             help='file-name or session-directory to target (default = .)')

    main_parser.add_argument('--debug',
                             required=False, action='store_true', default=False,
                             help='force all messaging to level debug')

    main_parser.add_argument('--silent',
                             required=False, action='store_true', default=False,
                             help='force no console messaging (including --debug)')

    main_parser.add_argument('--nolog',
                             required=False, action='store_true', default=False,
                             help='force no logfie creation (including --debug)')

    # parse command-line arguments

    arguments = main_parser.parse_args()

    # setup logging

    global debug_mode
    debug_mode = arguments.debug

    global log
    log = logging.getLogger(__name__)
    _logging_setup(arguments.task, arguments.target_type, arguments.target,
                   arguments.nolog, arguments.silent, debug_mode)
    log.debug('--- TEST of log.debug ---')
    log.info('--- TEST of log.info ---')
    log.warning('--- TEST of log.warning ---')
    log.error('--- TEST of log.error ---')
    log.critical('--- TEST of log.critical ---')

    log.info(os.path.basename(sys.argv[0]) + ' started.')
    log.debug(arguments)

    # prepare to execute task

    result = False

    # execute task

    #    task = arguments.task + '_' + arguments.target_type
    #    log.info('Executing: ' + task)
    #    result = locals()[task]()
    #    log.info('Result:    ' + str(result))

    if arguments.target_type == 'motor':
        if arguments.task == 'process':
            result = process_motor(arguments.target)
        elif arguments.task == 'report':
            result = report_motor(arguments.target)
        elif arguments.task == 'bundle':
            result = bundle_motor(arguments.target)

    if arguments.target_type == 'session':
        if arguments.task == 'process':
            result = process_session(arguments.target)
        elif arguments.task == 'report':
            result = report_session(arguments.target)
        elif arguments.task == 'bundle':
            result = bundle_session(arguments.target)

    # cleanup and exit

    log.info(os.path.basename(sys.argv[0]) + ' complete.')
    return result

#
# MoM (am I called as Main or as a Module)
#

if __name__ == '__main__':
    main()
