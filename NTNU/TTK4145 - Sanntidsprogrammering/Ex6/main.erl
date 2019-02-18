
-module(main).
-export([process_pairs_init/0]).

process_pairs_init() ->
  spawn(fun() -> process_pairs(0) end).

process_pairs(Num) ->
  Backup = spawn(fun() -> process_backup(Num) end),
  Kill = 10,
  count(Num, Backup, Kill+Num).

count(Num, Backup, Kill) ->
  if
    Num>Kill ->
      exit("");
    true ->
      io:fwrite("\n")
  end,
  Backup ! {ok,Num},                   %Send (Num) to Backup-process
  io:fwrite("Number: ~p\n", [Num]),
  timer:sleep(500),
  count(Num+1,Backup,Kill).

process_backup(Num) ->
  receive
    {ok,Num} ->
      process_backup(Num+1)
  after 5000 ->
    io:fwrite("Process died, backup taking over\n"),
    process_pairs(Num)
  end.
