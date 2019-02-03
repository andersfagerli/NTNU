defmodule Udp do
  @_SERVER_PORT   8973                #For local testing
  @_SEND_PORT     8972
  @_LOCAL_RECV    8971                #For local testing
  @_BROADCAST     {255,255,255,255}
  @_SERVER_IP     {10,100,23,242}
  @_LOCAL_IP      {127,0,0,1}

  def main_udp do
    {:ok,recvSocket} = :gen_udp.open(@_LOCAL_RECV)
    :inet.setopts(recvSocket, [{:active, false}])

    {:ok,sendSocket} = :gen_udp.open(@_SEND_PORT)
    :inet.setopts(sendSocket, [{:active, false}])

    spawn fn -> receive_udp_from_server(recvSocket) end
    spawn fn -> send_udp_to_server(sendSocket) end

  end

  def receive_udp_from_server(recvSocket) do
    {:ok, {address,port,packet}} = :gen_udp.recv(recvSocket,0)
    IO.puts("Received messsage: #{inspect(packet)} from server")
    receive_udp_from_server(recvSocket)
  end

  def send_udp_to_server(sendSocket) do
    :gen_udp.send(sendSocket, @_LOCAL_IP, @_SERVER_PORT, "Group 2 sending")
    :timer.sleep(10000)
    send_udp_to_server(sendSocket)
  end

end
