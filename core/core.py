from core.transmission.EdgeCommunicator import EdgeCommunicator

# 负责整体节点的调度

class Observer:
    def __init__(self, config_file):
        self.communicator = EdgeCommunicator(config_file)
        self.vehicle_id = 1

    def State_Updae(self, time_seq):