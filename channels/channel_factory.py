"""
channel factory
"""


def create_channel(channel_type):
    if channel_type == 'gradio':
        from channels.gradio.channel import GradioChannel
        return GradioChannel()
    raise RuntimeError
