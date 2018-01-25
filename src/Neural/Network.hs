{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
module Neural.Network
    ( Network(..)
      , NetOutput
      , module Neural.Layer
    ) where
import Neural.Layer
import Data.Array.Repa.Size

data Network :: Size -> [*] -> * where
    OutputLayer :: (Layer input layer)
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
    randomized = OutputLayer <$> randomized

instance (Randomized layer, Layer input layer, Randomized (Network (OutputSize input layer) (hidden ': layers)))
  => Randomized (Network input (layer ': hidden ': layers)) where
    randomized = (:~>) <$> randomized <*> randomized

instance (Show layer) => Show (Network input '[layer]) where
    show (OutputLayer layer) = show layer

instance (Show layer, Show (Network (OutputSize input layer) (hidden ': layers)))
  => Show (Network input (layer ': hidden ': layers)) where
    show (layer :~> network) = show layer ++ "\n~> " ++ show network

