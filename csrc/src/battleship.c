#include "../include/battleship.h"
#include "../include/placement.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

BATTLESHIP_API GameState *create_game(int height, int width, int num_ships,
                                      int *ship_lengths) {
  GameState *game = (GameState *)malloc(sizeof(GameState));
  if (!game)
    return NULL;

  game->height = height;
  game->width = width;
  game->num_ships = num_ships;

  
  game->board = (int32_t *)malloc(height * width * sizeof(int32_t));
  game->hits = (bool *)malloc(height * width * sizeof(bool));
  game->misses = (bool *)malloc(height * width * sizeof(bool));

  
  game->ship_lengths = (int *)malloc(num_ships * sizeof(int));
  memcpy(game->ship_lengths, ship_lengths, num_ships * sizeof(int));

  game->ship_sunk = (bool *)malloc(num_ships * sizeof(bool));

  return game;
}

BATTLESHIP_API void free_game(GameState *game) {
  if (game) {
    if (game->board)
      free(game->board);
    if (game->hits)
      free(game->hits);
    if (game->misses)
      free(game->misses);
    if (game->ship_lengths)
      free(game->ship_lengths);
    if (game->ship_sunk)
      free(game->ship_sunk);
    free(game);
  }
}

BATTLESHIP_API void reset_game(GameState *game, unsigned int seed) {
  
  for (int i = 0; i < game->height * game->width; i++) {
    game->board[i] = -1;
    game->hits[i] = false;
    game->misses[i] = false;
  }

  
  for (int i = 0; i < game->num_ships; i++) {
    game->ship_sunk[i] = false;
  }

  game->steps = 0;

  
  
  
  place_ships_random(game, seed);
}

BATTLESHIP_API void place_ship_fixed(GameState *game, int ship_id, int r, int c,
                                     int orientation) {
  
  int len = game->ship_lengths[ship_id];
  if (orientation == 0) { 
    for (int k = 0; k < len; k++) {
      game->board[r * game->width + (c + k)] = ship_id;
    }
  } else { 
    for (int k = 0; k < len; k++) {
      game->board[(r + k) * game->width + c] = ship_id;
    }
  }
}

static bool check_sunk(GameState *game, int ship_id) {
  
  bool all_hit = true;
  for (int i = 0; i < game->height * game->width; i++) {
    if (game->board[i] == ship_id) {
      if (!game->hits[i]) {
        all_hit = false;
        break;
      }
    }
  }
  return all_hit;
}

BATTLESHIP_API int step_game(GameState *game, int action) {
  if (action < 0 || action >= game->height * game->width) {
    return -1; 
  }

  
  if (game->hits[action] || game->misses[action]) {
    return -1; 
  }

  int ship_id = game->board[action];
  game->steps++;

  if (ship_id != -1) {
    
    game->hits[action] = true;

    
    if (check_sunk(game, ship_id)) {
      game->ship_sunk[ship_id] = true;
      return 2; 
    }
    return 1; 
  } else {
    
    game->misses[action] = true;
    return 0; 
  }
}

BATTLESHIP_API void get_observation(GameState *game, float *buffer) {
  
  
  
  

  int size = game->height * game->width;
  for (int i = 0; i < size; i++) {
    bool is_hit = game->hits[i];
    bool is_miss = game->misses[i];

    float val_hit = is_hit ? 1.0f : 0.0f;
    float val_miss = is_miss ? 1.0f : 0.0f;
    float val_unknown = (!is_hit && !is_miss) ? 1.0f : 0.0f;

    buffer[0 * size + i] = val_hit;
    buffer[1 * size + i] = val_miss;
    buffer[2 * size + i] = val_unknown;
  }
}
