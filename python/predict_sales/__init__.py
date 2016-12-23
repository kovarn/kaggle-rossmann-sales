import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('../rossman.log')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
ch_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

fh.setFormatter(fh_formatter)
ch.setFormatter(ch_formatter)

logger.addHandler(ch)
logger.addHandler(fh)
