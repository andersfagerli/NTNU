%Server IP: 10.100.23.242
%129.241.231.44

-module(main).
-export([main_udp/0]).

-define(SEND_PORT,20000+2). %Server port for lab station 2
-define(RECV_PORT,8790).

main_udp() ->
  {ok,Socket} = gen_udp:open(?RECV_PORT, [binary, {active,false}]),
  %{ok,SendSocket} = gen_udp:open(?SEND_PORT, [binary, {active,false}]),

  spawn(fun() -> receive_from_server(Socket) end),
  spawn(fun() -> send_to_server(Socket) end).

receive_from_server(Socket) ->
  gen_udp:recv(Socket,0),
  receive_from_server(Socket).

send_to_server(Socket) ->
  ok = gen_udp:send(Socket, {10,100,23,242}, ?SEND_PORT, "Test"),
  timer:sleep(5000),
  send_to_server(Socket).
