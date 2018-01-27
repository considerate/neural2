{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Neural.Network
    ( Network(..)
      , NetOutput
      , CastSize(..)
      , module Neural.Layer
    ) where
import Neural.Layer
import Data.Array.Repa.Size
import Data.Serialize

-- | layer1 :~> layer2 :~> OutputLayer layer3
data Network :: Size -> [*] -> * where
    OutputLayer :: (Layer input layer)
      => layer
      -> Network input '[layer]
    (:~>) :: (Layer input layer)
          => layer
          -> Network (OutputSize input layer) (hidden ': layers)
          -> Network input (layer ': hidden ': layers)
infixr 6 :~>

instance (Layer input layer) => Serialize (Network input '[layer]) where
    put (OutputLayer layer) = put layer
    get = do
        layer <- get
        pure (OutputLayer layer)

instance (Layer input layer, Serialize (Network (OutputSize input layer) (hidden ': layers)))
  => Serialize (Network input (layer ': hidden ': layers)) where
    put (layer :~> layers) = do
        put layer
        put layers
    get = do
        layer <- get
        layers <- get
        pure (layer :~> layers)

type family NetOutput (net :: *) :: Size
type instance NetOutput (Network input '[layer]) = OutputSize input layer
type instance NetOutput (Network input (layer ': hidden ': layers)) = NetOutput (Network (OutputSize input layer) (hidden ': layers))

instance (Randomized layer, Layer input layer)
  => Randomized (Network input '[layer]) where
    randomized = OutputLayer <$> randomized

instance (Randomized layer, Layer input layer, Randomized (Network (OutputSize input layer) (hidden ': layers)))
  => Randomized (Network input (layer ': hidden ': layers)) where
    randomized = (:~>) <$> randomized <*> randomized

instance (Show layer) => Show (Network input '[layer]) where
    show (OutputLayer layer) = show layer

instance (Show layer, Show (Network (OutputSize input layer) (hidden ': layers)))
  => Show (Network input (layer ': hidden ': layers)) where
    show (layer :~> network) = show layer ++ "\n~> " ++ show network

class CastSize a b where
    cast :: a -> b

instance (Layer input1 layer, Layer input2 layer)
  => CastSize (Network input1 '[layer]) (Network input2 '[layer]) where
    cast (OutputLayer layer) = OutputLayer layer

instance (Layer input1 layer
  , Layer input2 layer
  , CastSize (Network (OutputSize input1 layer) (hidden ': layers)) (Network (OutputSize input2 layer) (hidden ': layers))
         )
    => CastSize (Network input1 (layer ': hidden ': layers)) (Network input2 (layer ': hidden ': layers)) where
        cast (layer :~> rest) = layer :~> cast rest

-- instance (Layer input layer) => Updatable (Network input '[layer]) where
--     update params net grad =

-- instance (Layer input layer) => Layer input (Network input '[layer]) where
--
