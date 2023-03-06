/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * NITRO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#include <import/nitf.h>

NITF_CXX_GUARD


static const char* tres[] = {
    #include "tre_config.h"
    NULL
};

#define MAX_IDENT 1000
static const char* ident[MAX_IDENT] = {
    NITF_PLUGIN_TRE_KEY,
    NULL
};

static nitf_TREHandler DEFAULT_TRESHandler; 
NITFAPI(const char**) DEFAULT_TRES_init(nitf_Error* error){ 
    nitf_DLL* dll;
    dll = nitf_DLL_construct(error);
    nitf_DLL_load(dll, "DEFAULT_TRES", error);

    int identIdx = 1;

    for (int i = 0; tres[i]; ++i)
    {
        NITF_PLUGIN_INIT_FUNCTION init;
        const char** subIdent;
        
        char name[NITF_MAX_PATH];
        memset(name, 0, NITF_MAX_PATH);
        NITF_SNPRINTF(name, NITF_MAX_PATH, "%s%s", tres[i], NITF_PLUGIN_INIT_SUFFIX);
        init = (NITF_PLUGIN_INIT_FUNCTION)nitf_DLL_retrieve(dll, name, error);
        if (!init)
        {
            nitf_Error_print(error, stdout, "Invalid init hook in DSO");
            return NULL;
        }
        
        /*  Else, call it  */
        
        subIdent = (*init)(error);
        if (!subIdent)
        {
            nitf_Error_initf(error,
                NITF_CTXT,
                NITF_ERR_INVALID_OBJECT,
                "The plugin [%s] is not retrievable",
                tres[i]);
            return NULL;
        }

        /* Check for ID */
        
        if (strcmp(subIdent[0], NITF_PLUGIN_TRE_KEY) != 0)
        {
            nitf_Error_initf(error,
                NITF_CTXT,
                NITF_ERR_INVALID_OBJECT,
                "The plugin [%s] is not retrievable",
                tres[i]);
            continue;
        }

        /* Add to master Ident list */
        
        for (int s = 1; subIdent[s]; ++s)
        {
            if (identIdx >= MAX_IDENT)
            {
                nitf_Error_initf(error,
                    NITF_CTXT,
                    NITF_ERR_INVALID_OBJECT,
                    "Too Many TREs, unable to register more than 1000",
                    tres[i]);
                return NULL;
            }
        
            ident[identIdx++] = subIdent[s];
        }
    }

    ident[identIdx++] = NULL;

    return ident; 
} 
NITFAPI(nitf_TREHandler*) DEFAULT_TRES_handler(nitf_Error* error) { 
    (void)error; 
    return &DEFAULT_TRESHandler; 
}

NITF_CXX_ENDGUARD
