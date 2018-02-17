{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module Neural.Layers.ReLU where
import Prelude(pure, ($), (<), (*), otherwise, Num, Ord,Show)
import Data.Serialize
import Neural.Layer
import Data.Array.Repa.SizedArray

newtype ReLU = ReLU ()
    deriving(Show)

instance Serialize ReLU where
    put _ = put ()
    get = pure (ReLU ())

instance Randomized ReLU where
    randomized = pure (ReLU ())

instance Updatable ReLU where
    type Gradient ReLU = ()
    update _ layer _ = layer

relu :: (Num p, Ord p) => p -> p
relu x
  | x < 0  = 0
  | otherwise = x

relu' :: (Num p, Ord p) => p -> p
relu' x
  | x < 0  = 0
  | otherwise = 1

instance (Sized input) => Layer input ReLU where
    type OutputSize input ReLU = input
    forward _ x = computeP $ map relu x
    backward _ x _ dy = do
        dx <- computeP $ zipWith (*) (map relu' x) dy
        pure (dx, ())

