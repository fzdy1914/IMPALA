import tarfile
import os.path

with tarfile.open('submission.tar.gz', "w:gz") as tar:
    tar.add('agent/DenseNetAgent.py', arcname="main.py")
    tar.add('state/our_100000_after_start_model.pt', arcname="model.pt")
    tar.add('model.py', arcname="model.py")
    tar.add('parameters.py', arcname="parameters.py")
    tar.add('board_stack_plus.py', arcname="board_stack_plus.py")
    tar.add('silent_agent_helper.py', arcname="silent_agent_helper.py")