{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where
import Text.Read(readMaybe)
import System.Environment(getArgs)
import Neural.Layers
import Data.Array.Repa.SizedArray
import Neural.Network
import Neural.Training
import qualified Prelude
import Prelude (Double,IO,(-),(/),(+),(*),(.),(<=),putStrLn, pure, fmap,print,Int,fromInteger)
import Control.Monad(replicateM)
import Control.Monad.Random(MonadRandom,getRandomR)
import Data.Maybe(fromMaybe,Maybe(..))
import Data.List
import GHC.TypeLits
import Data.Proxy

inCircle :: (Double, Double) -> Double
inCircle (x,y) = if inside then 1 else 0
    where
        circles = [((0.5, 0.4), 0.5), ((-0.8,-0.5),0.5)]
        inside = Prelude.or (fmap f circles)
        f ((cx,cy),r) = ((x - cx)*(x - cx) + (y + cy)*(y + cy)) <= r*r

type TestNet n = Network ('ZZ '::. n '::. 2) [Weights 2 16, Logistic, Weights 16 8, Logistic, Weights 8 1, Logistic]

initializeNet :: (KnownNat n, MonadRandom m) => m (TestNet n)
initializeNet = randomized

screen :: [[(Double, Double)]]
screen = [ [ (x / 25 - 1, y / 10 - 1) | x <- [0..50] ] | y <- [0..20] ]

inputs :: [[SizedArray U ('ZZ '::. 1 '::. 2)]]
inputs = fmap (fmap (\(x,y) -> fromList [x,y])) screen

getPrediction :: KnownNat n => SizedArray U (NetOutput (TestNet n)) -> Double
getPrediction x = x ! (Z :. 0 :. 0)

getCoord :: MonadRandom m => m (Double, Double)
getCoord = do
    x <- getRandomR (-1, 1)
    y <- getRandomR (-1, 1)
    pure (x,y)

chunks :: Int -> [a] -> [[a]]
chunks n = Prelude.takeWhile (Prelude.not . Prelude.null) . Data.List.unfoldr (Just . splitAt n)

netTest :: Prelude.Int -> IO ()
netTest n = do
    coords <- replicateM n getCoord
    let targets = fmap inCircle coords :: [Double]
    let xs = chunks (fromInteger batchsize * 2) (concat (fmap pairList coords)) :: [[Double]]
    let ts = chunks (fromInteger batchsize) targets
    case someNatVal batchsize of
      Just (SomeNat (Proxy :: Proxy batchsize)) -> do
            let !samples = Prelude.zipWith toArrays xs ts
            network <- initializeNet :: IO (TestNet batchsize)
            trained <- trainMany (Params 0.5) network samples
            let predictNet = cast trained :: TestNet 1
            outputs <- (Prelude.traverse . Prelude.traverse) (prediction predictNet) inputs
            let rendered = (fmap . fmap) (render . getPrediction) outputs
            putStrLn (Prelude.unlines ((fmap . fmap) (render . inCircle) screen))
            putStrLn (Prelude.unlines rendered)
    where
        batchsize = 1
        pairList (x,y) = [x,y]
        toArrays xs ts = (fromList xs, fromList ts)
        prediction net point = predict point net
        render r
            | r <= 0.2  = ' '
            | r <= 0.4  = '.'
            | r <= 0.6  = '-'
            | r <= 0.8  = '='
            | Prelude.otherwise = '#'

getMaybe :: Prelude.Int -> [a] -> Prelude.Maybe a
getMaybe _ [] = Prelude.Nothing
getMaybe 0 (x:_) = Prelude.Just x
getMaybe n (_:xs) = getMaybe (n-1) xs

main :: IO ()
main = do
    args <- getArgs
    let n = getMaybe 0 args Prelude.>>= readMaybe
    putStrLn "Training"
    netTest (fromMaybe 1000 n)
    putStrLn "done"
    print ()
