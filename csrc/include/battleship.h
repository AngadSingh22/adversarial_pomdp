#ifndef BATTLESHIP_H
#define BATTLESHIP_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
  int height;
  int width;
  int32_t *board; 
  bool *hits;     
  bool *misses;   

  int num_ships;
  int *ship_lengths; 
  bool *ship_sunk;   

  int steps; 
} GameState;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define BATTLESHIP_API __declspec(dllexport)
#else
#define BATTLESHIP_API
#endif

BATTLESHIP_API GameState *create_game(int height, int width, int num_ships,
                                      int *ship_lengths);
BATTLESHIP_API void free_game(GameState *game);

BATTLESHIP_API void reset_game(GameState *game,
                               unsigned int seed); 
BATTLESHIP_API void
place_ship_fixed(GameState *game, int ship_id, int r, int c,
                 int orientation); 

BATTLESHIP_API int step_game(GameState *game, int action);

BATTLESHIP_API void get_observation(GameState *game, float *buffer);

#ifdef __cplusplus
}
#endif

#endif 
