# encoding:utf-8

from channels.channel_factory import create_channel
from common.log import logger

channel = create_channel('gradio')
demo = channel.server

if __name__ == '__main__':
    try:
        # startup channel
        logger.info('App starting up...')
        channel.startup()
    except Exception as e:
        logger.error('App startup failed!')
        logger.exception(e)
