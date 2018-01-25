{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
module Data.Array.Repa.SizedArray
    ( (|*|)
      -- , (#>)
      -- , (<#)
      , (^+)
      , (^-)
      , (^*)
      , (^/)
      , (!)
      -- , (><)
      , expand
      , reshape
      , zipWith
      , map
      , computeP
      , computeS
      , sumP
      , fromFunction
      , fromList
      , SizedArray
      , transpose
      , deepSeqArray
      , Source
      , U
      , D
      , Repa.Z(..)
      , (:.)(..)
      , Load
      , Target
      , DIM0
      , DIM1
      , DIM2
      , DIM3
      , module Data.Array.Repa.Size
    )
where

import Data.Array.Repa.Size
import qualified Prelude
import Prelude (Monad,Double,Int,(-),(/),(+),(*),($),(<$>),(.),(++),otherwise,error,Show,show,length,take)
import qualified Data.Array.Repa as Repa
import qualified Data.Array.Repa.Unsafe as Repa
import qualified Data.Array.Repa.Eval as Repa
import Data.Array.Repa(Source,Array,U,D,(:.)(..),All(..),Any(..), DIM0, DIM1, DIM2, DIM3)
import Data.Array.Repa.Eval(Load,Target)
import GHC.TypeLits
import Data.Proxy

newtype SizedArray r (size :: Size) = SA {getArray :: Array r (ShapeOf size) Double}

instance forall size. Sized size => Show (SizedArray U size) where
    show (SA x) = show x

instance forall size. Sized size => Show (SizedArray D size) where
    show (SA x) = show (Repa.computeS x :: Array U (ShapeOf size) Double) -- need to evaluate delayed computation before showing


transpose :: (Sized size, Source r Double)
          => SizedArray r (size '::. n '::. m)
          -> SizedArray D (size '::. m '::. n)
transpose (SA x) = SA $ Repa.transpose x

deepSeqArray :: (Sized size, Source r Double) => SizedArray r size -> b -> b
deepSeqArray (SA x) = Repa.deepSeqArray x

sumP :: (ShapeOf size2 ~ (ShapeOf size1 :. Int), Monad f, Source r Double, Repa.Shape (ShapeOf size1)) => SizedArray r size2 -> f (SizedArray U size1)
sumP (SA x) = SA <$> Repa.sumP x

computeP :: (Sized size
  , Source r2 Double
  , Monad m
  , Load r (ShapeOf size) Double
  , Target r2 Double
  )
  => SizedArray r size
  -> m (SizedArray r2 size)
computeP (SA x) = SA <$> Repa.computeP x

computeS :: (Sized size, Load r (ShapeOf size) Double, Target r2 Double)
         => SizedArray r size
         -> SizedArray r2 size
computeS = SA . Repa.computeS . getArray

map :: (Sized size, Source r Double) => (Double -> Double) -> SizedArray r size -> SizedArray D size
map f (SA x) = SA $ Repa.map f x

fromFunction :: forall size. (Sized size) => (ShapeOf size -> Double) -> SizedArray D size
fromFunction element = SA $ Repa.fromFunction shape element
    where
        shape = shapeOf (Proxy :: Proxy size)

lengthAtLeast :: (Prelude.Num t, Prelude.Eq t) => t -> [a] -> Prelude.Bool
lengthAtLeast 0 _ = Prelude.True
lengthAtLeast 1 [] = Prelude.False
lengthAtLeast n (_:xs) = lengthAtLeast (n-1) xs

fromList :: forall size. (Sized size) => [Double] -> SizedArray U size
fromList xs
  | lengthAtLeast volume xs = SA $ Repa.fromList shape (take volume xs)
  | otherwise = error
        $ "List passed to SizedArray.fromList is too short. Requires length of at least "
        ++ show volume
        ++ " got size of "
        ++ show (length xs)
    where
        shape = shapeOf (Proxy :: Proxy size)
        volume = Repa.size shape

zipWith :: (Sized size, Source r1 Double, Source r2 Double)
        => (Double -> Double -> Double)
        -> SizedArray r1 size
        -> SizedArray r2 size
        -> SizedArray D size
zipWith f (SA x) (SA y) = SA $ Repa.zipWith f x y

(^+), (^-), (^*), (^/) :: (Sized size, Source r1 Double, Source r2 Double)
     => SizedArray r1 size
     -> SizedArray r2 size
     -> SizedArray D size
(^+) = zipWith (+)
(^-) = zipWith (-)
(^*) = zipWith (*)
(^/) = zipWith (/)

reshape :: forall size1 size2 r.
    (Sized size1, Sized size2, Source r Double, Volume size1 ~ Volume size2)
      => SizedArray r size1
      -> SizedArray D size2
reshape (SA x) = SA y
    where
        shape = shapeOf (Proxy :: Proxy size2)
        y = Repa.reshape shape x

expand :: forall small big r. (Sized small, Sized big, (ShapeOf small) `SubShape` (ShapeOf big), Source r Double)
       => SizedArray r small
       -> SizedArray D big
expand (SA x) = x `Repa.deepSeqArray` SA $ Repa.unsafeBackpermute shape subShape x
    where
        shape = shapeOf (Proxy :: Proxy big)


mmultP  :: (Repa.Shape sh, Source r1 Double, Source r2 Double)
        => Array r1 (sh :. Int :. Int) Double
        -> Array r2 (sh :. Int :. Int) Double
        -> Array D (sh :. Int :. Int) Double
mmultP arr brr
  = arr `Repa.deepSeqArray` brr `Repa.deepSeqArray` do
      let transposed = Repa.transpose brr
      Repa.fromFunction (outer :. h1 :. w2)
        $ \(_ :. row :. col) ->
          Repa.sumAllS
          $ Repa.zipWith (*)
                (Repa.unsafeSlice arr (Any :. row :. All))
                (Repa.unsafeSlice transposed (Any :. col :. All))
      where
          (outer :. h1 :. _) = Repa.extent arr
          (_ :. _ :. w2) = Repa.extent brr
(|*|) :: (Sized size, KnownNat a, KnownNat c, Source r1 Double, Source r2 Double)
      => SizedArray r1 (size '::. a '::. b)
      -> SizedArray r2 (size '::. b '::. c)
      -> SizedArray D (size '::. a '::. c)
(SA xs) |*| (SA ys) = SA $ (mmultP xs ys)
infixl 6 |*|


(!) :: (Sized size, Source r Double) => SizedArray r size -> (ShapeOf size) -> Double
(!) (SA x) shape = x Repa.! shape

-- (<#) :: forall x y r1 r2.
--     (KnownNat x, KnownNat y, Source r1 Double, Source r2 Double)
--       => SizedArray r1 ('ZZ '::. x)
--       -> SizedArray r2 ('ZZ '::. x '::. y)
--       -> SizedArray D ('ZZ '::. y)
-- (SA vec) <# (SA mat) = fromFunction f
--     where
--         f (_ :. col) = Repa.sumAllS zs
--             where
--                 zs = Repa.zipWith (*) vec (Repa.unsafeSlice mat (Any :. All :. col))
-- -- vec <# mat = reshape (vec' |*| mat)
-- --     where
-- --         vec' = reshape vec :: SizedArray D ('ZZ '::. 1 '::. x)
--
-- (#>) :: forall x y r1 r2.
--     (KnownNat x, KnownNat y, Source r1 Double, Source r2 Double)
--       => SizedArray r2 ('ZZ '::. x '::. y)
--       -> SizedArray r1 ('ZZ '::. y)
--       -> SizedArray D ('ZZ '::. x)
-- mat #> vec = reshape (mat |*| vec')
--     where
--         vec' = reshape vec :: SizedArray D ('ZZ '::. y '::. 1)
--
-- (><) :: forall n m r1 r2.
--     (KnownNat n, KnownNat m, Source r1 Double, Source r2 Double)
--       => SizedArray r1 ('ZZ '::. n)
--       -> SizedArray r2 ('ZZ '::. m)
--       -> SizedArray D ('ZZ '::. n '::. m)
-- x >< y = x' |*| y'
--     where
--         x' = reshape x :: SizedArray D ('ZZ '::. n '::. 1)
--         y' = reshape y :: SizedArray D ('ZZ '::. 1 '::. m)
