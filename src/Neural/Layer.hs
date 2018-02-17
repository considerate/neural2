{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
module Neural.Layer where
import Data.Array.Repa.SizedArray
import Control.Monad.Random
import Data.Serialize

newtype LearningParameters = Params Double
newtype Grad a = Grad a

-- | Randomly initialize layer
class Randomized layer where
    randomized :: MonadRandom m => m layer

-- | Update a layer using `LearningParameters` and layer `Gradient`
class (Serialize layer) => Updatable layer where
    type Gradient layer :: *
    update :: LearningParameters -> layer -> Gradient layer -> layer

-- | Layer is defined by forward and backward operations as well as update
class (Updatable layer, Sized input, Sized (OutputSize input layer))
  => Layer (input :: Size) layer where
    type OutputSize input layer :: Size

    -- | Forward pass for layer
    forward :: (Monad m)
            => layer
            -> SizedArray U input
            -> m (SizedArray U (OutputSize input layer))
    -- | Back-propagation pass for layer
    backward :: (Monad m)
             => layer
             -> SizedArray U input -- forward pass input
             -> SizedArray U (OutputSize input layer) -- forward pass output
             -> SizedArray U (OutputSize input layer) -- propagated gradient on output
             -> m (SizedArray U input, Gradient layer)

