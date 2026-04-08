function DF = get_can_data()
global ch
%% Receive Messages
% Receive all available messages from the channel.

 % msgs = receive(ch, Inf, "OutputFormat", "timetable");
% load("msgs.mat")

%% Repackage the Signal Data
% Convert received messages into individual timetables. The |canSignalTimetable| 
% function returns a structure with one field for each unique message in the timetable. 
% Each field value is a timetable of all the signals defined in that message.

% if ~isempty(msgs)

 % sigs = canSignalTimetable(msgs);
load("sigs.mat")
s = sigs.Speed_L.Speed_Left;
TIME = seconds(sigs.Joystick.Time);
JS = table2array(sigs.Joystick);



SPL = s(1:2:length(s));
s = sigs.Speed_R.Speed_Right;
SPR = s(1:2:length(s));

a=length(TIME);
b=length(SPL);
c=length(SPR);
d = min([a b c]);

if a>d 
    TIME=TIME(1:d);
    JS=JS(1:d,:);
end
if b>d
    SPL=SPL(1:d);
end
if c>d
    SPR=SPR(1:d);
end

DF = [TIME JS SPR SPL];
% else
%     DF=[];
% end
end