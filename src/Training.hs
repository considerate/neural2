{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
module Training where
import qualified Prelude
import Prelude(Double, pure, ($))
import Size
import SizedArray
import Control.Monad
import Network
import Layer

predict :: (Monad m)
        => SizedArray U input
        -> Network input layers
        -> m (SizedArray U (NetOutput (Network input layers)))
predict x (O layer) = forward layer x
predict x (layer :~> layers) = do
    y <- forward layer x
    predict y layers

backprop :: (Monad m, Source r Double, net ~ Network i layers)
         => SizedArray U i
         -> SizedArray r (NetOutput net)
         -> net
         -> m (SizedArray U i, net)
backprop x t (O layer)
    = do
    y <- forward layer x
    dy <- computeP (y ^- t)
    (dx, dWeights) <- backward layer x y dy
    pure (dx, O $ update (Params 0.25) layer dWeights)
backprop x t (layer :~> rest) = do
    y <- forward layer x
    (dy, rest') <- backprop y t rest
    (dx, dWeights) <- backward layer x y dy
    pure (dx, (update (Params 0.25) layer dWeights) :~> rest')

trainOne :: (Sized input, net ~ Network input layers, output ~ NetOutput net, Monad m)
         => (SizedArray U input, SizedArray U output)
         -> net
         -> m net
trainOne (x, t) net = fmap Prelude.snd (backprop x t net)

trainMany :: (Sized input, net ~ Network input layers, output ~ NetOutput net, Monad m)
         => [(SizedArray U input, SizedArray U output)]
         -> net
         -> m net
trainMany samples network = foldM (Prelude.flip trainOne) network samples
