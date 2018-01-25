{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Neural.Layers.Weights where
import qualified Prelude
import Prelude(Double, pure, ($), (-), (*))
import Neural.Layer
import GHC.TypeLits
import Data.Array.Repa.SizedArray
import Control.Monad
import Control.Monad.Random(MonadRandom,getRandomRs)

data Weights (i :: Nat) (o :: Nat)
    = Weights {layerWeights :: !(SizedArray U ('ZZ '::. i '::. o)), layerBias :: !(SizedArray U ('ZZ '::. o))}
    deriving (Prelude.Show)


randomWeights :: (KnownNat i, KnownNat o, MonadRandom m) => m (Weights i o)
randomWeights = do
    ws <- getRandomRs (-1,1)
    b <- getRandomRs (-1,1)
    pure $ Weights (fromList ws) (fromList b)

instance (KnownNat i, KnownNat o) => Updatable (Weights i o) where
    type Gradient (Weights i o) = Weights i o
    update (Params rate) (Weights ws bs) (Weights dws dbs)
      = Weights
        (computeS $ zipWith (\w dw -> w - rate * dw) ws dws)
        (computeS $ zipWith (\b db -> b - rate * db) bs dbs)

instance (KnownNat i, KnownNat o) => Randomized (Weights i o) where
    randomized = randomWeights

sumBatch :: (Sized size, Monad m, Source r Double) => SizedArray r (size '::. n '::. o) -> m (SizedArray U (size '::. o))
sumBatch x = sumP (transpose x)

instance (KnownNat i, KnownNat o, KnownNat n, input ~ ('ZZ '::. n '::. i), output ~ ('ZZ '::. n '::.  o)) => Layer ('ZZ '::. n '::. i) (Weights i o) where
    type OutputSize ('ZZ '::. n '::. i) (Weights i o) = 'ZZ '::. n '::. o

    forward (Weights ws b) x = computeP $ (x |*| ws) ^+ expand b

    backward (Weights ws _) x _ dy = do
        dw <- computeP $ transpose x |*| dy
        dx <- computeP $ dy |*| transpose ws
        db <- sumBatch dy
        pure (dx, Weights dw db)
            where
