#ifndef PLACEMENT_H
#define PLACEMENT_H

#include "battleship.h" 

#ifdef __cplusplus
extern "C" {
#endif

int place_ships_random(GameState *game, unsigned int seed);

int is_valid_placement(const GameState *game, int ship_id, int r, int c,
                       int orientation);

#ifdef __cplusplus
}
#endif

#endif 
