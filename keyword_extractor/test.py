import sys
sys.path.append("../")
import utils.textrank as textrank

print(textrank.preprocess_text('''This a re-port of a perl interface to Tk8.4.
C code is derived from Tcl/Tk8.4.5.
It also includes all the C code parts of Tix8.1.4 from SourceForge.
The perl code corresponding to Tix's Tcl code is not fully implemented.

Perl API is essentially the same as Tk800 series Tk800.025 but has not
been verified as compliant. There ARE differences see pod/804delta.pod.","
Perl Graphical User Interface ToolKit."'''))