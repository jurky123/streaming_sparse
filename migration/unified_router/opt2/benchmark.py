"""Reuse the unchanged measurement harness with an explicit adapter replacement."""
from .. import fast_bench
from .network import load_optimized

def main():
    fast_bench.load_fast=load_optimized
    fast_bench.main()

if __name__=='__main__':main()
