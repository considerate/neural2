{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE StandaloneDeriving #-}
module Neural.Layers.Convolution where

import Prelude(Double, fromInteger, ($), (-), otherwise, (<), (>=), (+), (*), Bool(..), (<$>), (<*>), pure, Show, const, sqrt, (/), fromIntegral)
import qualified Data.Array.Repa as Repa
import qualified Data.Array.Repa.Unsafe as Repa
import Data.Array.Repa.SizedArray
import Data.Proxy
import GHC.TypeLits
import GHC.TypeLits.Extra()
import Data.Serialize
import Neural.Layer
import Control.Monad.Random(getRandomRs)

-- assuming a stride of 1
correlation :: forall weightSize inputSize outputSize batches kernels channels
    kernelHeight kernelWidth height width resultHeight resultWidth r1 r2.
    (weightSize ~ ('ZZ '::. kernels '::. channels '::. kernelHeight '::. kernelWidth)
      , inputSize ~ ('ZZ '::. batches '::. channels '::. height '::. width)
      , outputSize ~ ('ZZ '::. batches '::. kernels '::. resultHeight '::. resultWidth)
      , Sized weightSize
      , Sized inputSize
      , Sized outputSize
      , Source r1 Double
      , Source r2 Double
      , KnownNat kernels
      , KnownNat kernelWidth
      , KnownNat kernelHeight
      , height ~ (kernelHeight + resultHeight - 1)
      , width ~ (kernelWidth + resultWidth - 1)
    )
    => SizedArray r1 weightSize
    -> SizedArray r2 inputSize
    -> SizedArray D outputSize
correlation (SA weights) (SA xs) = fromFunction convolve
    where
        kernelHeight = fromInteger $ natVal (Proxy :: Proxy kernelHeight)
        kernelWidth = fromInteger $ natVal (Proxy :: Proxy kernelWidth)
        kernels = fromInteger $ natVal (Proxy :: Proxy kernels)
        convolve (Z :. n :. k :. y :. x) = Repa.sumAllS (kernel Repa.*^ image)
            where
                -- get kernel k from weights
                kernel = Repa.extract (Z :. k :. 0 :. 0 :. 0) (Z :. 1 :. kernels :. kernelHeight :. kernelWidth) weights
                -- get 3d image (with all channels) for current sample (n) of kernel size
                -- from (x,y) to (x+kernelWidht,y+kernelHeight)
                image = Repa.extract (Z :. n :. 0 :. y :. x) (Z :. 1 :. kernels :. kernelHeight :. kernelWidth) xs

rotateWeights :: forall kernels channels kernelHeight kernelWidth r.
    (KnownNat kernels, KnownNat channels, KnownNat kernelHeight, KnownNat kernelWidth, Source r Double)
      => SizedArray r ('ZZ '::. kernels '::. channels '::. kernelHeight '::. kernelWidth)
      -> SizedArray D ('ZZ '::. channels '::. kernels '::. kernelHeight '::. kernelWidth)
rotateWeights weights = backpermute rotation weights
    where
        kernelHeight = fromInteger $ natVal (Proxy :: Proxy kernelHeight)
        kernelWidth = fromInteger $ natVal (Proxy :: Proxy kernelWidth)
        rotation (Z :. channel :. kernel :. y :. x) = Z :. kernel :. channel :. y' :. x'
            where
                -- rotated by 180 degrees
                y' = kernelHeight - y - 1
                -- rotated by 180 degrees
                x' = kernelWidth - x - 1

zeroPad :: forall outer height width height' width' r widthPadding heightPadding size padded.
    (size ~ (outer '::. width '::. height)
      , padded ~ (outer '::. width' '::. height')
      , Sized size
      , Sized padded
      , heightPadding ~ Half (height' - height)
      , widthPadding ~ Half (width' - width)
      , Source r Double
      , KnownNat height
      , KnownNat width
      , KnownNat heightPadding
      , KnownNat widthPadding
    )
    => SizedArray r size
    -> SizedArray D padded
zeroPad xs = traverse xs pad
    where
        height = fromInteger $ natVal (Proxy :: Proxy height)
        width = fromInteger $ natVal (Proxy :: Proxy width)
        heightPadding = fromInteger $ natVal (Proxy :: Proxy heightPadding)
        widthPadding = fromInteger $ natVal (Proxy :: Proxy widthPadding)
        outside (y, x)
            | y < heightPadding = True
            | y >= height + heightPadding = True
            | x < widthPadding = True
            | x >= width + widthPadding = True
            | otherwise = False
        pad lookup (outer :. py :. px)
          | outside (py, px) = 0 -- all values are zero in padding
          | otherwise = lookup (outer :. (py-heightPadding) :. (px-widthPadding)) -- get values from original image


data Convolution (channels :: Nat) (kernels :: Nat) (kernelHeight :: Nat) (kernelWidth :: Nat) (resultHeight :: Nat) (resultWidth :: Nat) where
    Convolution :: (KnownNat kernels, KnownNat channels, KnownNat kernelHeight, KnownNat kernelWidth, KnownNat resultHeight, KnownNat resultWidth)
                => SizedArray U ('ZZ '::. kernels '::. channels '::. kernelHeight '::. kernelWidth) -- weights
                -> SizedArray U ('ZZ '::. kernels '::. resultHeight '::. resultWidth) -- bias
                -> Convolution channels kernels kernelHeight kernelWidth resultWidth resultHeight
deriving instance Show (Convolution c k kh kw rw rh)

instance (KnownNat kernels, KnownNat channels, KnownNat kernelHeight, KnownNat kernelWidth, KnownNat resultHeight, KnownNat resultWidth)
  => Serialize (Convolution channels kernels kernelHeight kernelWidth resultWidth resultHeight) where
      put (Convolution ws b) = do
          put ws
          put b
      get = Convolution <$> get <*> get

instance forall kernels channels kernelHeight kernelWidth resultHeight resultWidth.
    (KnownNat kernels
      , KnownNat channels
      , KnownNat kernelHeight
      , KnownNat kernelWidth
      , KnownNat resultHeight
      , KnownNat resultWidth
    )
  => Randomized (Convolution channels kernels kernelHeight kernelWidth resultHeight resultWidth) where
      randomized = do
        ws <- getRandomRs (- r, r)
        pure $ Convolution (fromList ws) (computeS $ fromFunction (const 0))
            where
                kernels = natVal (Proxy :: Proxy kernels)
                -- channels = natVal (Proxy :: Proxy channels)
                -- kernelWidth = natVal (Proxy :: Proxy kernelWidth)
                -- kernelHeight = natVal (Proxy :: Proxy kernelHeight)
                resultHeight = natVal (Proxy :: Proxy resultHeight)
                resultWidth = natVal (Proxy :: Proxy resultWidth)
                n = fromIntegral $ kernels * resultWidth * resultHeight
                r = 1 / (sqrt n)
                -- r = (sqrt 6) / (sqrt n_in + sqrt n_out)

instance (KnownNat kernels, KnownNat channels, KnownNat kernelHeight, KnownNat kernelWidth, KnownNat resultHeight, KnownNat resultWidth)
  => Updatable (Convolution channels kernels kernelHeight kernelWidth resultWidth resultHeight) where
      type Gradient
        (Convolution channels kernels kernelHeight kernelWidth resultWidth resultHeight)
        = Grad (Convolution channels kernels kernelHeight kernelWidth resultWidth resultHeight)
      update (Params rate) (Convolution ws bs) (Grad (Convolution dws dbs))
        = Convolution
        (computeS $ zipWith (\w dw -> w - rate * dw) ws dws)
        (computeS $ zipWith (\b db -> b - rate * db) bs dbs)

convolution :: forall kernels channels kernelHeight kernelWidth height width resultHeight resultWidth batches r1 r2.
    ( KnownNat channels
      , KnownNat kernels
      , KnownNat kernelHeight
      , KnownNat kernelWidth
      , KnownNat height
      , KnownNat width
      , KnownNat resultHeight
      , KnownNat resultWidth
      , KnownNat batches
      , resultHeight ~ (kernelHeight + height - 1)
      , resultWidth ~ (kernelWidth + width - 1)
      , (kernelHeight + resultHeight - 1) ~ (height + (2 * (kernelHeight - 1)))
      , (kernelWidth + resultWidth - 1) ~ (width + (2 * (kernelWidth - 1)))
      , KnownNat (height + (2 * (kernelHeight - 1)))
      , KnownNat (width + (2 * (kernelWidth - 1)))
      , KnownNat (Half (kernelWidth + resultWidth - 1 - width))
      , KnownNat (Half (kernelHeight + resultHeight - 1 - height))
      , Source r1 Double
      , Source r2 Double
    )
    => SizedArray r1 ('ZZ '::. channels '::. kernels '::. kernelHeight '::. kernelWidth)
    -> SizedArray r2 ('ZZ '::. batches '::. channels '::. height '::. width)
    -> SizedArray D ('ZZ '::. batches '::. kernels '::. resultHeight '::. resultWidth)
convolution weights xs
  = correlation kernels images
        where
            kernels = rotateWeights weights :: SizedArray D ('ZZ '::. kernels '::. channels '::. kernelHeight '::. kernelWidth)
            images = zeroPad xs

correlationWeights :: forall kernels channels kernelHeight kernelWidth batches width height resultHeight resultWidth r1 r2.
  ( KnownNat batches
    , KnownNat kernels
    , KnownNat kernelHeight
    , KnownNat kernelWidth
    , KnownNat resultHeight
    , KnownNat resultWidth
    , KnownNat channels
    , Source r1 Double
    , Source r2 Double
  )
  => SizedArray r1 ('ZZ '::. batches '::. kernels '::. resultHeight '::. resultWidth)
  -> SizedArray r2 ('ZZ '::. batches '::. channels '::. height '::. width)
  -> SizedArray D ('ZZ '::. kernels '::. channels '::. kernelHeight '::. kernelWidth)
correlationWeights (SA dy) (SA xs) = fromFunction convolve
    where
        batches = fromInteger $ natVal (Proxy :: Proxy batches)
        kernelWidth = fromInteger $ natVal (Proxy :: Proxy kernelWidth)
        kernelHeight = fromInteger $ natVal (Proxy :: Proxy kernelHeight)
        convolve (Z :. k :. c :. a :. b) = Repa.sumAllS (kernel Repa.*^ image)
            where
                kernel = Repa.extract (Z :. 0 :. k :. 0 :. 0) (Z :. batches:. 1 :. kernelHeight :. kernelWidth) dy
                image = Repa.extract (Z :. 0 :. c :. a :. b) (Z :. batches :. 1 :. kernelHeight :. kernelWidth) xs

sumBatch' :: (Source r Double, KnownNat batches, KnownNat channels, KnownNat resultHeight, KnownNat resultWidth)
          => SizedArray r ('ZZ '::. batches '::. channels '::. resultHeight '::. resultWidth)
          -> SizedArray D ('ZZ '::. channels '::. resultHeight '::. resultWidth)
sumBatch' (SA xs) = fromFunction sumRow
    where
        sumRow (Z :. c :. y :. x) = Repa.sumAllS $ Repa.unsafeSlice xs (Repa.Any :. c :. y :. x)

type family Half (n :: Nat) :: Nat where
    Half 0 = 0
    Half n = Half (n-2) + 1

-- Assuming stride = 1
-- Assuming padding = 1
-- resultWidth = (width - kernelWidth + 2P)/S +1
-- resultWidth = width - kernelWidth + 2 + 1
-- resultWidth = width - kernelWidth + 3
-- width = resultWidth + kernelWidth - 3
instance (KnownNat kernels
  , KnownNat channels
  , KnownNat kernelHeight
  , KnownNat kernelWidth
  , KnownNat resultHeight
  , KnownNat resultWidth
  , KnownNat batches
  , KnownNat width
  , KnownNat height
  , width ~ (kernelWidth + resultWidth - 1)
  , height ~ (kernelHeight + resultHeight - 1)
  , (kernelHeight + height - 1) ~ (resultHeight + (2 * (kernelHeight - 1)))
  , (kernelWidth + width - 1) ~ (resultWidth + (2 * (kernelWidth - 1)))
  , KnownNat (resultHeight + (2 * (kernelHeight - 1)))
  , KnownNat (resultWidth + (2 * (kernelWidth - 1)))
  , KnownNat (Half (kernelWidth + width - 1 - resultWidth))
  , KnownNat (Half (kernelHeight + height - 1 - resultHeight))
  )
    => Layer
    ('ZZ '::. batches '::. channels '::. height '::. width)
    (Convolution channels kernels kernelHeight kernelWidth resultWidth resultHeight) where
        type OutputSize
            ('ZZ '::. batches '::. channels '::. height '::. width)
            (Convolution channels kernels kernelHeight kernelWidth resultWidth resultHeight)
          = ('ZZ '::. batches '::. kernels '::. resultHeight '::. resultWidth)

        forward (Convolution ws b) x = computeP $ (correlation ws x) ^+ expand b

        backward (Convolution ws _) x _ dy = do
            dx <- computeP $ ws `convolution` dy
            dw <- computeP $ dy `correlationWeights` x
            db <- computeP $ sumBatch' dy
            pure (dx, Grad $ Convolution dw db)


