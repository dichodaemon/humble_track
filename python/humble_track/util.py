def flatten( features ):
  return features.reshape( [features.shape[0] * features.shape[1], features.shape[2]] )

