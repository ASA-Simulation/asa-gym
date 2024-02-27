from google.protobuf.any_pb2 import Any
from google.protobuf.message import Message
from zmq.sugar.socket import Socket

import asagym.proto.simulator_pb2 as pb


def send_message_to_simulation(socket: Socket, message: Message) -> None:
    """Sends the response to the simulator.

    Blocking communication pattern. The message is generic.
    """
    # multiplex the message into Any
    any_request = Any()
    any_request.Pack(message)

    # send message through zmq socket
    request_message = pb.RequestMessage(payload=any_request)
    socket.send(request_message.SerializeToString())


def recv_message_from_simulation(
    socket: Socket, msg_type: pb.INIT | pb.CLOSE | pb.STEP | pb.RESET
) -> pb.InitResponse | pb.ResetResponse | pb.StepResponse | pb.CloseResponse:
    """Waits for the response from the simulator and demultiplexes it according to given type.

    Blocking communication pattern. The response is generic, matching the type argument.
    """

    # receive message through zmq socket
    reply_message = pb.ResponseMessage()
    reply_message.ParseFromString(socket.recv())

    # demultiplex the message according to expected type
    if msg_type == pb.INIT:
        reply = pb.InitResponse()
    elif msg_type == pb.RESET:
        reply = pb.ResetResponse()
    elif msg_type == pb.STEP:
        reply = pb.StepResponse()
    elif msg_type == pb.CLOSE:
        reply = pb.CloseResponse()
    else:
        raise Exception(
            msg_type,
            "not valid, should be one of: pb.INIT | pb.CLOSE | pb.STEP | pb.RESET",
        )

    # unpack the message with expected type
    any_reply = reply_message.payload
    any_reply.Unpack(reply)
    return reply
