import rospy
class TextColors:
  """
  The C{TextColors} class is used as alternative to the C{rospy} logger. It's useful to
  print messages when C{roscore} is not running.
  """
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  log_level = rospy.INFO

  def disable(self):
    """
    Resets the coloring.
    """
    self.HEADER = ''
    self.OKBLUE = ''
    self.OKGREEN = ''
    self.WARNING = ''
    self.FAIL = ''
    self.ENDC = ''

  def blue(self, msg):
    """
    Prints a B{blue} color message
    @type  msg: string
    @param msg: the message to be printed.
    """
    print((self.OKBLUE + msg + self.ENDC))

  def debug(self, msg):
    """
    Prints a B{green} color message
    @type  msg: string
    @param msg: the message to be printed.
    """
    print((self.OKGREEN + msg + self.ENDC))

  def error(self, msg):
    """
    Prints a B{red} color message
    @type  msg: string
    @param msg: the message to be printed.
    """
    print((self.FAIL + msg + self.ENDC))

  def ok(self, msg):
    """
    Prints a B{green} color message
    @type  msg: string
    @param msg: the message to be printed.
    """
    print((self.OKGREEN + msg + self.ENDC))

  def warning(self, msg):
    """
    Prints a B{yellow} color message
    @type  msg: string
    @param msg: the message to be printed.
    """
    print((self.WARNING + msg + self.ENDC))

  def logdebug(self, msg):
    """
    Prints message with the word 'Debug' in green at the begging.
    Alternative to C{rospy.logdebug}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= rospy.DEBUG:
      print((self.OKGREEN + 'Debug ' + self.ENDC + str(msg)))

  def loginfo(self, msg):
    """
    Prints message with the word 'INFO' begging.
    Alternative to C{rospy.loginfo}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= rospy.INFO:
      print(('INFO ' + str(msg)))

  def logwarn(self, msg):
    """
    Prints message with the word 'Warning' in yellow at the begging.
    Alternative to C{rospy.logwarn}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= rospy.WARN:
      print((self.WARNING + 'Warning ' + self.ENDC + str(msg)))

  def logerr(self, msg):
    """
    Prints message with the word 'Error' in red at the begging.
    Alternative to C{rospy.logerr}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= rospy.ERROR:
      print((self.FAIL + 'Error ' + self.ENDC + str(msg)))

  def logfatal(self, msg):
    """
    Prints message with the word 'Fatal' in red at the begging.
    Alternative to C{rospy.logfatal}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= rospy.FATAL:
      print((self.FAIL + 'Fatal ' + self.ENDC + str(msg)))

  def set_log_level(self, level):
    """
    Sets the log level. Possible values are:
      - DEBUG:  1
      - INFO:   2
      - WARN:   4
      - ERROR:  8
      - FATAL:  16
    @type  level: int
    @param level: the new log level
    """
    self.log_level = level