# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


_BANNER = """
***********************************************************************************************************************************************
{0:s} ▄▄▄▄▄▄▄▄▄▄▄ {1:s} ▄▄▄▄▄▄▄▄▄▄▄{2:s}  ▄▄▄▄▄▄▄▄▄▄▄{3:s}  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄
{0:s}▐░░░░░░░░░░░▌{1:s}▐░░░░░░░░░░░▌{2:s}▐░░░░░░░░░░░▌{3:s}▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
{0:s} ▀▀▀▀█░█▀▀▀▀{1:s} ▐░█▀▀▀▀▀▀▀█░▌{2:s} ▀▀▀▀█░█▀▀▀▀{3:s} ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀█░▌
{0:s}     ▐░▌    {1:s} ▐░▌       ▐░▌{2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌
{0:s}     ▐░▌    {1:s} ▐░█▄▄▄▄▄▄▄█░▌{2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌
{0:s}     ▐░▌    {1:s} ▐░░░░░░░░░░░▌{2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
{0:s}     ▐░▌    {1:s} ▐░█▀▀▀▀█░█▀▀ {2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀█░█▀▀
{0:s}     ▐░▌    {1:s} ▐░▌     ▐░▌  {2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌     ▐░▌
{0:s}     ▐░▌    {1:s} ▐░▌      ▐░▌ {2:s} ▄▄▄▄█░█▄▄▄▄{3:s} ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌      ▐░▌
{0:s}     ▐░▌    {1:s} ▐░▌       ▐░▌{2:s}▐░░░░░░░░░░░▌{3:s}▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌
{0:s}      ▀     {1:s}  ▀         ▀ {2:s} ▀▀▀▀▀▀▀▀▀▀▀ {3:s} ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀

Viva la révolution!

A DASK distributed RFI flagger by Science Data Processing and Radio Astronomy Research Group
Copyright 2019 South African Radio Astronomy Observatory (SARAO, SKA-SA)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

***********************************************************************************************************************************************
"""  # noqa


def banner():
    RED = '\033[0;31m'
    WHITE = '\033[0;37m'
    BLUE = '\033[0;34m'
    RESET = '\033[0m'

    # make it Frenchy
    return _BANNER.format(BLUE, WHITE, RED, RESET)
