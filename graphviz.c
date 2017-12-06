/** @file     graphviz.c
 *  @brief    Graphviz's dot format output.
 *  @author   Brian K. (AlpacaRotorvator@users.noreply.github.com)
 *  @date     10/18/2017
 */

#include "graphviz.h"
#include <stdio.h>

void gen_graphviz (float ** c, int * m, int n) {
    FILE * outfile = fopen("minpath.dot", "w");

    fprintf(outfile, "digraph minpath {\n\
  dpi = 300;\n\
  node [shape=\"point\", width=0.075];\n\
  edge [arrowhead=\"none\", penwidth=1];\n\
  overlap=\"scale\";\n\
  splines=\"true\";\n");

    for(int i = 0; i < n; i++) {
	fprintf(outfile, "  \"%d\" [pos=\"%f,%f!\"];\n", i, c[i][0], c[i][1]);
    }

    for(int i = 0; i < n - 1; i++) {
	fprintf(outfile, "  \"%d\" -> \"%d\";\n", m[i], m[i + 1]);
    }
    
    fprintf(outfile, "  \"%d\" -> \"%d\";\n", m[n - 1], m[0]);
    
    fprintf(outfile, "}");

    fclose(outfile);
}
