%% AeroCortex MATLAB Digital Twin Interface Client
% Team AB16 - Amrita Vishwa Vidyapeetham
% Streams UAV state bus to AeroCortex API gateway and applies cognitive recovery commands.

clear; clc;
fprintf('=== AeroCortex UAV MATLAB Digital Twin Client ===\n');

api_url = 'http://127.0.0.1:8000/api/v1/telemetry';
webopts = weboptions('MediaType', 'application/json', 'Timeout', 5);

% Mission Simulation Parameters
dt = 0.1; % 100ms step
num_steps = 100;

% Initial State
uav_state.x = 0.0;
uav_state.y = 0.0;
uav_state.z = 45.0;
uav_state.vx = 0.0;
uav_state.vy = 8.0;
uav_state.vz = 0.0;
uav_state.roll = 0.0;
uav_state.pitch = 2.0;
uav_state.yaw = 90.0;
uav_state.battery = 95.0;
uav_state.gps_sats = 12;
uav_state.hdop = 0.9;
uav_state.gps_valid = true;
uav_state.rssi = -60.0;
uav_state.comms_ok = true;
uav_state.wind_speed = 4.0;
uav_state.wind_heading = 45.0;
uav_state.mode = 'MISSION_NAV';

fprintf('Connecting to AeroCortex API Gateway at %s...\n', api_url);

for step = 1:num_steps
    % Inject a simulated GPS failure at step 30
    if step == 30
        fprintf('[Simulink Event] Injecting GPS Denial / Multipath Interference!\n');
        uav_state.gps_sats = 2;
        uav_state.hdop = 5.2;
        uav_state.gps_valid = false;
    end
    
    % Update simulation time and battery
    uav_state.timestamp = posixtime(datetime('now'));
    uav_state.battery = max(0.0, uav_state.battery - 0.04);
    
    try
        % Transmit telemetry packet to AeroCortex
        response = webwrite(api_url, uav_state, webopts);
        
        % Parse command
        action_name = response.action;
        is_reactive = response.is_reactive;
        
        fprintf('Step %d: Sent telemetry -> Drone Cmd: [%s] (Reactive: %d)\n', ...
            step, action_name, is_reactive);
        
        % In a real Simulink model, map action_name to control loops:
        if strcmp(action_name, 'SWITCH_INERTIAL_DEAD_RECKONING')
            uav_state.mode = 'INERTIAL_DEAD_RECKONING';
            uav_state.vy = 4.0; % Reduce speed
        elseif strcmp(action_name, 'ALTITUDE_HOLD_DESCENT')
            uav_state.mode = 'ALT_HOLD_LOW_WIND';
            uav_state.z = max(20.0, uav_state.z - 0.5);
        elseif strcmp(action_name, 'CONTROLLED_EMERGENCY_LAND')
            uav_state.mode = 'EMERGENCY_LAND';
            uav_state.z = max(0.0, uav_state.z - 0.3);
        end
        
    catch ME
        fprintf('Error communicating with AeroCortex API: %s\n', ME.message);
        break;
    end
    
    pause(dt);
end

fprintf('Simulation run completed.\n');
