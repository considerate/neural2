{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Neural.Layers.MaxPooling where
import GHC.TypeLits
import Data.Serialize
import Neural.Layer
import Data.Array.Repa.SizedArray
import GHC.TypeLits.Extra(Div)
import Data.Proxy

import Prelude(pure,($),fromInteger,maximum,Int,(+),(-),otherwise,div,(==))
newtype MaxPooling (n :: Nat) = MaxPooling ()

instance Serialize (MaxPooling n) where
    put _ = put ()
    get = pure (MaxPooling ())

instance Randomized (MaxPooling n) where
    randomized = pure (MaxPooling ())

instance Updatable (MaxPooling n) where
    type Gradient (MaxPooling n) = ()
    update _ layer _ = layer

instance forall batches channels height width poolSize.
  (KnownNat batches, KnownNat channels, KnownNat height, KnownNat width, KnownNat poolSize, KnownNat (Div height poolSize), KnownNat (Div width poolSize))
  => Layer ('ZZ '::. batches '::. channels '::. height '::. width) (MaxPooling poolSize) where
      type OutputSize ('ZZ '::. batches '::. channels '::. height '::. width) (MaxPooling poolSize)
        = 'ZZ '::. batches '::. channels '::. (Div height poolSize) '::. (Div width poolSize)

      forward _ xs = computeP $ traverse xs maxPool
        where
            poolSize = fromInteger $ natVal (Proxy :: Proxy poolSize) :: Int
            maxPool lookup (outer :. y :. x) = maximum [lookup (outer :. (y + dy) :. (x + dx))  | dx <- [0..(poolSize-1)], dy <- [0..(poolSize-1)]]

      backward _ x y dy = do
          dx <- computeP $ fromFunction getMax
          pure (dx, ())
              where
                  poolSize = fromInteger $ natVal (Proxy :: Proxy poolSize) :: Int
                  outputPos (outer :. row :. col) = outer :. (row `div` poolSize) :. (col `div` poolSize)
                  isMax pos = (x ! pos) == (y ! outputPos pos)
                  getMax pos
                    | isMax pos = dy ! outputPos pos
                    | otherwise = 0

