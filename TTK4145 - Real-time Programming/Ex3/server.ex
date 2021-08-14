defmodule Server do
  @_PORT        8973
  @_SENDER_PORT 8971  #For local testing, sender port known

  def server_init do
    {:ok,serverSocket} = :gen_udp.open(@_PORT)
    :inet.setopts(serverSocket, [{:active, false}])

    server_respond(serverSocket)
  end

  def server_respond(serverSocket) do
    {:ok, {address,port,packet}} = :gen_udp.recv(serverSocket,0)
    IO.puts("Received message: #{inspect(packet)} from IP: #{inspect(address)} on port #{inspect(port)}")
    :gen_udp.send(serverSocket, address, @_SENDER_PORT, "Server received")

    server_respond(serverSocket)
  end

end
