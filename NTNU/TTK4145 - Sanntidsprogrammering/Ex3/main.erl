%Server IP: 10.100.23.242
%129.241.231.44

-module(main).
-export([main/0]).

-define(SEND_PORT,20000+2). %Server port for lab station 2
-define(RECV_PORT,8790).

main() ->
  {ok,RecvSocket} = gen_udp:open(?RECV_PORT, [binary, {active,false}]),
  {ok,SendSocket} = gen_udp:open(?SEND_PORT, [binary, {active,false}]),

  spawn(fun() -> receive_from_server(RecvSocket) end),
  spawn(fun() -> send_to_server(SendSocket) end).

receive_from_server(Socket) ->
  gen_udp:recv(Socket,0),
  receive_from_server(Socket).

send_to_server(Socket) ->
  ok = gen_udp:send(Socket, {10,100,23,242}, ?SEND_PORT, "Test"),
  timer:sleep(5000),
  send_to_server(Socket).
