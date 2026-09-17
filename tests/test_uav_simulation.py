import pytest
from aerocortex.simulator.uav_dynamics import UAVDigitalTwin

def test_uav_digital_twin_nominal_flight():
    uav = UAVDigitalTwin(mission_id="SIM-TEST-01")
    initial_bat = uav.battery_pct
    initial_y = uav.y
    
    # Run 10 steps
    for _ in range(10):
        tele = uav.step(dt=0.1)
        
    assert tele.y > initial_y # Drone moved forward
    assert tele.battery_pct < initial_bat # Battery drained
    assert tele.gps_valid is True
    assert tele.comms_connected is True

def test_uav_failure_injection_and_recovery():
    uav = UAVDigitalTwin(mission_id="SIM-TEST-02")
    
    # Inject GPS loss
    uav.inject_failure("gps_loss", True)
    tele = uav.step(dt=0.1)
    assert tele.gps_valid is False
    assert tele.gps_satellites <= 4
    
    # Command recovery: ALTITUDE_HOLD_DESCENT
    uav.set_recovery_action("ALTITUDE_HOLD_DESCENT")
    initial_z = uav.z
    for _ in range(15):
        tele = uav.step(dt=0.1)
    assert tele.z < initial_z # Descended safely

def test_uav_emergency_landing():
    uav = UAVDigitalTwin(mission_id="SIM-TEST-03")
    uav.set_recovery_action("CONTROLLED_EMERGENCY_LAND")
    
    # Run until landed
    for _ in range(100):
        tele = uav.step(dt=0.5)
        if uav.mission_status == "SAFE_RECOVERY_LANDED":
            break
            
    assert uav.mission_status == "SAFE_RECOVERY_LANDED"
    assert tele.z <= 0.5
