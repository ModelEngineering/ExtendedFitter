import pandas as pd
import time

class Logger():

    def __init__(self):
        self.startTime = time.time()

    def _write(self, msg, numNL):
        relTime = time.time() - self.startTime
        newLineStr = ('').join(["\n" for _ in range(numNL)])
        newMsg = "\n%s%f: %s" % (newLineStr, relTime, msg)
        print(newMsg)

    def error(self, msg, excp):
        # Progress message
        fullMsg = "%s: %s" % (msg, str(excp))
        self._write("    (%s)" % fullMsg, 0)
