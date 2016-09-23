#!/usr/bin/env python
# coding: utf-8
from ete3 import Tree

tree =  Tree("((දුම්වැටි,යුද්ධය)මරණය,(සුන්දර,නර්තනයක්)පිරිසිදු);", format=1)
print tree.get_ascii(show_internal=True)