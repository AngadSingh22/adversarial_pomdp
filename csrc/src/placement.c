#include "../include/placement.h"
#include "../include/battleship.h"
#include <stdlib.h>
#include <string.h>

static void _shuffle_int(int *arr, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

int is_valid_placement(const GameState *game, int ship_id, int r, int c,
                       int orientation) {
  int len = game->ship_lengths[ship_id];

  if (orientation == 0) { 
    if (c + len > game->width)
      return 0;
    for (int k = 0; k < len; k++) {
      if (game->board[r * game->width + (c + k)] != -1)
        return 0;
    }
  } else { 
    if (r + len > game->height)
      return 0;
    for (int k = 0; k < len; k++) {
      if (game->board[(r + k) * game->width + c] != -1)
        return 0;
    }
  }
  return 1;
}

int place_ships_random(GameState *game, unsigned int seed) {
  srand(seed);

  int area = game->height * game->width;

  
  int max_candidates = 2 * area;
  int *cand_r   = (int *)malloc(max_candidates * sizeof(int));
  int *cand_c   = (int *)malloc(max_candidates * sizeof(int));
  int *cand_ori = (int *)malloc(max_candidates * sizeof(int));
  int *order    = (int *)malloc(max_candidates * sizeof(int));

  if (!cand_r || !cand_c || !cand_ori || !order) {
    free(cand_r); free(cand_c); free(cand_ori); free(order);
    return 0; 
  }

  int success = 1;

  for (int ship_id = 0; ship_id < game->num_ships; ship_id++) {
    
    int n = 0;

    
    for (int r = 0; r < game->height; r++) {
      for (int c = 0; c < game->width; c++) {
        if (is_valid_placement(game, ship_id, r, c, 0)) {
          cand_r[n]   = r;
          cand_c[n]   = c;
          cand_ori[n] = 0;
          n++;
        }
      }
    }
    
    for (int r = 0; r < game->height; r++) {
      for (int c = 0; c < game->width; c++) {
        if (is_valid_placement(game, ship_id, r, c, 1)) {
          cand_r[n]   = r;
          cand_c[n]   = c;
          cand_ori[n] = 1;
          n++;
        }
      }
    }

    if (n == 0) {
      success = 0;
      break;
    }

    
    for (int i = 0; i < n; i++) order[i] = i;
    _shuffle_int(order, n);

    int idx = order[0];
    int r   = cand_r[idx];
    int c   = cand_c[idx];
    int ori = cand_ori[idx];
    int len = game->ship_lengths[ship_id];

    
    if (ori == 0) { 
      for (int k = 0; k < len; k++)
        game->board[r * game->width + (c + k)] = ship_id;
    } else { 
      for (int k = 0; k < len; k++)
        game->board[(r + k) * game->width + c] = ship_id;
    }
  }

  free(cand_r);
  free(cand_c);
  free(cand_ori);
  free(order);
  return success;
}
