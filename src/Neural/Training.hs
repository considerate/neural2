{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Neural.Training where
import qualified Prelude
import Prelude(Double, pure, ($))
import Data.Array.Repa.SizedArray
import Control.Monad
import Neural.Network
import GHC.TypeLits
import Data.Proxy()

predict :: (Monad m)
        => SizedArray U input
        -> Network input layers
        -> m (SizedArray U (NetOutput (Network input layers)))
predict x (OutputLayer layer) = forward layer x
predict x (layer :~> layers) = do
    y <- forward layer x
    predict y layers

backprop :: forall m r net i layers n. (Monad m, Source r Double, net ~ Network i layers, n ~ Volume (NetOutput net), KnownNat n)
         => LearningParameters
         -> SizedArray U i
         -> SizedArray r (NetOutput net)
         -> net
         -> m (SizedArray U i, net)
backprop params x t (OutputLayer layer)
    = do
    y <- forward layer x
    dy <- computeP $ (y ^- t)
    (dx, dWeights) <- backward layer x y dy
    pure (dx, OutputLayer $ update params layer dWeights)
        -- where
            -- n = fromIntegral $ natVal (Proxy :: Proxy n)
backprop params x t (layer :~> rest) = do
    y <- forward layer x
    (dy, rest') <- backprop params y t rest
    (dx, dWeights) <- backward layer x y dy
    pure (dx, (update params layer dWeights) :~> rest')

trainOne :: (Sized input
  , net ~ Network input layers
  , output ~ NetOutput net
  , KnownNat (Volume output)
  , Monad m)
         => LearningParameters
         -> net
         -> (SizedArray U input, SizedArray U output)
         -> m net
trainOne params net (x, t) = fmap Prelude.snd (backprop params x t net)

trainMany :: (Sized input
  , net ~ Network input layers
  , output ~ NetOutput net
  , KnownNat (Volume output)
  , Monad m)
         => LearningParameters
         -> net
         -> [(SizedArray U input, SizedArray U output)]
         -> m net
trainMany params = foldM (trainOne params)
