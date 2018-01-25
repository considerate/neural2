{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Neural.Layers.Logistic where
import Prelude(Floating, Num, Show, (/), (+), (*), (-), ($), pure, exp)
import Neural.Layer
import Data.Array.Repa.SizedArray

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp(-x))

sigmoid' :: Num a => a -> a
sigmoid' y = y * (1 - y)

newtype Logistic = Logistic ()
    deriving (Show)
instance Randomized Logistic where
    randomized = pure $ Logistic ()
instance Updatable Logistic where
    type Gradient Logistic = ()
    update _ layer _ = layer

instance Sized input => Layer input Logistic where
    type OutputSize input Logistic = input
    forward _ x = computeP $ map sigmoid x
    backward _ _ y dy = do
        dx <- computeP $ (map sigmoid' y) ^* dy
        pure (dx, ())
