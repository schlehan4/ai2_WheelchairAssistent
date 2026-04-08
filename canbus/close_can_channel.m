function close_can_channel()
global ch db
%% Stop the Channel
% Disconnect the channel.

stop(ch)

%% Clean Up
% Clear unneeded variables.

clear ch db
end