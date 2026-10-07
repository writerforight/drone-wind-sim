/**
 * modes/ — every game or lab mode is a module with the same small interface, registered here.
 *
 *   { id, name, description, camera: 'chase' | 'orbit', panels: [...],
 *     start(app),                 // put the drone where the mode begins
 *     reference(app, sticks) }    // the controller reference for this tick (from the sticks or a plan)
 *
 * A new mode (race, waypoints, a learned policy) is a new file plus one line in this list.
 */
import { freeFlight } from './free-flight.js';
import { tuningLab } from './tuning-lab.js';

export const modes = [freeFlight, tuningLab];
