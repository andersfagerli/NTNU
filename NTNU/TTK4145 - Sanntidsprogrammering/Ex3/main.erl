%Server IP: 10.100.23.242
%129.241.231.44

-module(main).
-export([main_udp/0, main_tcp/0]).

-define(SERVER_UDP_PORT,20000+2). %Server udp port for lab station 2
-define(SERVER_TCP_PORT,33546).   %Server tcp port
-define(SEND_PORT,8791).




main_udp() ->
  {ok,RecvSocket} = gen_udp:open(?SERVER_UDP_PORT, [list, {active,false}]),
  {ok,SendSocket} = gen_udp:open(?SEND_PORT, [binary, {active,false}]),

  spawn(fun() -> receive_udp_from_server(RecvSocket) end),
  spawn(fun() -> send_udp_to_server(SendSocket) end).

receive_udp_from_server(RecvSocket) ->
  {ok, {Address, Port, Packet}} = gen_udp:recv(RecvSocket,0),
  io:fwrite("Receiving\n"), %How to print message from server??
  receive_udp_from_server(RecvSocket).

send_udp_to_server(SendSocket) ->
  ok = gen_udp:send(SendSocket, {10,100,23,242}, ?SERVER_UDP_PORT, "Group 2 sending"),
  timer:sleep(5000),
  send_udp_to_server(SendSocket).





main_tcp() ->
  {ok, ListenSocket} = gen_tcp:listen(?SERVER_TCP_PORT, [{active,true}, binary]), %Correct port to listen to??
  {ok, SendSocket} = gen_tcp:connect({10,100,23,242}, ?SERVER_TCP_PORT, [binary, {active,true}]),
  gen_tcp:send(SendSocket, "Connect to: 129.241.231.44:8791\0"),

  spawn(fun() -> receive_tcp_from_server(ListenSocket) end),
  spawn(fun() -> send_tcp_to_server(SendSocket) end).

receive_tcp_from_server(ListenSocket) ->
  {ok, AcceptSocket} = gen_tcp:accept(ListenSocket), %How to receive message??
  io:fwrite("Receiving\n"),
  receive_tcp_from_server(ListenSocket).

send_tcp_to_server(SendSocket) ->
  gen_tcp:send(SendSocket, "Hello there\0"),
  timer:sleep(5000),
  send_tcp_to_server(SendSocket).
