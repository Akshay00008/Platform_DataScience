# Defining logs
import logging as lg

lg.basicConfig(filename='Logger.log', level=lg.INFO, format='%(asctime)s %(message)s')
logger = lg.getLogger()

class Logs:
    
    def Logging(self, msg):
        logger.info(msg)