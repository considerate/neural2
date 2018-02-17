{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-} -- GHC can't figure out that 'ZZ is a base case for Volume given the multiplication
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}
module Data.Array.Repa.Size where
import Prelude(Int,fromInteger,Show)
import Data.Array.Repa
import GHC.TypeLits
import Data.Proxy

-- | Repa shape (Z :. Int :. Int)

-- | Type-level size
data Size = ZZ | Size ::. Nat
infixl 3 ::.

-- data SomeSize = forall size. Sized size => SomeSize (Proxy size)
--
-- newtype SSize (size :: Size) = SSize Size
--
-- withSize :: (Sized size => Proxy size -> b)
--          -> SSize size -> Proxy size -> b
--

-- | Shape of type-level Size
type family ShapeOf (size :: Size) :: *
type instance ShapeOf 'ZZ = Z
type instance ShapeOf (outer '::. n) = ShapeOf outer :. Int

class (Show (ShapeOf size), Shape (ShapeOf size)) => Sized (size :: Size) where
    type Volume size :: Nat
    type PrependDimension (n :: Nat) size :: Size
    shapeOf :: f size -> ShapeOf size

instance Sized 'ZZ where
    type Volume 'ZZ = 1
    type PrependDimension n 'ZZ = 'ZZ '::. n
    shapeOf _ = Z

instance forall sz n. (Sized sz, KnownNat n) => Sized (sz '::. n) where
    type Volume (sz '::. n) = Volume sz * n
    type PrependDimension k (sz '::. n) = (PrependDimension k sz) '::. n
    shapeOf _ = outerShape :. n
        where
            outerShape = shapeOf (Proxy :: Proxy sz)
            n = fromInteger (natVal (Proxy :: Proxy n)) :: Int

class (Slice smaller, Slice larger) => smaller `SubShape` larger where
    subShape :: larger -> smaller
instance Slice larger => Z `SubShape` larger where
    subShape _ = Z
instance (smaller `SubShape` larger) => (smaller :. Int) `SubShape` (larger :. Int) where
    subShape (rest :. n) = (subShape rest :. n)

class (Sized smaller
  , Sized larger
  , ShapeOf smaller `SubShape` ShapeOf larger)
  => smaller `SubSize` larger
instance (Sized larger, Slice (ShapeOf larger)) => 'ZZ `SubSize` larger
instance (Sized smaller, Sized larger, smaller `SubSize` larger, KnownNat n) => (smaller '::. n) `SubSize` (larger '::. n)

