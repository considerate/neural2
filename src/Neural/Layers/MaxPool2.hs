{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}
module Neural.Layers.MaxPool2(MaxPool2(..)) where
import GHC.TypeLits
import Data.Serialize
import Neural.Layer
import Data.Array.Repa.SizedArray
import GHC.TypeLits.Extra()
import Data.Proxy()
import Prelude(pure,($),maximum,(+),(*),otherwise,div,(==),Show)

newtype MaxPool2 = MaxPool2 ()
    deriving (Show)

instance Serialize MaxPool2 where
    put _ = put ()
    get = pure (MaxPool2 ())

instance Randomized (MaxPool2) where
    randomized = pure (MaxPool2 ())

instance Updatable (MaxPool2) where
    type Gradient (MaxPool2) = ()
    update _ layer _ = layer

type family Half (n :: Nat) :: Nat where
    Half 0 = 0
    Half n = Half (n-2) + 1

instance forall batches channels height width.
  (KnownNat batches, KnownNat channels, KnownNat height, KnownNat width, KnownNat (Half height), KnownNat (Half width))
  => Layer ('ZZ '::. batches '::. channels '::. height '::. width) (MaxPool2) where
      type OutputSize ('ZZ '::. batches '::. channels '::. height '::. width) (MaxPool2)
        = 'ZZ '::. batches '::. channels '::. (Half height) '::. (Half width)

      forward _ xs = computeP $ traverse xs maxPool
        where
            maxPool lookup (outer :. y :. x) = maximum [lookup (outer :. (2*y + dy) :. (2*x + dx))  | dx <- [0..1], dy <- [0..1]]

      backward _ x y dy = do
          dx <- computeP $ fromFunction getMax
          pure (dx, ())
              where
                  -- poolSize = fromInteger $ natVal (Proxy :: Proxy poolSize) :: Int
                  outputPos (outer :. row :. col) = outer :. (row `div` 2) :. (col `div` 2)
                  isMax pos = (x ! pos) == (y ! outputPos pos)
                  getMax pos
                    | isMax pos = dy ! outputPos pos
                    | otherwise = 0

