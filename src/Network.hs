{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
module Network where
import Layer
import Size

data Network :: Size -> [*] -> * where
    O :: (Layer input layer)
      => layer
      -> Network input '[layer]
    (:~>) :: (Layer input layer)
          => layer
          -> Network (OutputSize input layer) (hidden ': layers)
          -> Network input (layer ': hidden ': layers)
infixr 6 :~>

type family NetOutput (net :: *) :: Size
type instance NetOutput (Network input '[layer]) = OutputSize input layer
type instance NetOutput (Network input (layer ': hidden ': layers)) = NetOutput (Network (OutputSize input layer) (hidden ': layers))

instance (Randomized layer, Layer input layer)
  => Randomized (Network input '[layer]) where
    randomized = O <$> randomized

instance (Randomized layer, Layer input layer, Randomized (Network (OutputSize input layer) (hidden ': layers)))
  => Randomized (Network input (layer ': hidden ': layers)) where
    randomized = (:~>) <$> randomized <*> randomized

instance (Show layer) => Show (Network input '[layer]) where
    show (O layer) = show layer

instance (Show layer, Show (Network (OutputSize input layer) (hidden ': layers)))
  => Show (Network input (layer ': hidden ': layers)) where
    show (layer :~> network) = show layer ++ "\n~> " ++ show network

