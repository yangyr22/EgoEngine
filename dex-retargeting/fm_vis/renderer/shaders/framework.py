
# Mario Rosasco, 2016
# adapted from framework.cpp, Copyright (C) 2010-2012 by Jason L. McKesson
# This file is licensed under the MIT License.
#
# NB: Unlike in the framework.cpp organization, the main loop is contained
# in the tutorial files, not in this framework file. Additionally, a copy of
# this module file must exist in the same directory as the tutorial files
# to be imported properly.

import os

try:
    from OpenGL.GL import *
except:
    pass


# Function that creates and compiles shaders according to the given type (a GL enum value) and
# shader program (a file containing a GLSL program).
def loadShader(shaderType, shaderFile):
    # check if file exists, get full path name
    strFilename = findFileOrThrow(shaderFile)
    shaderData = None
    with open(strFilename, 'r') as f:
        shaderData = f.read()

    shader = glCreateShader(shaderType)
    glShaderSource(shader, shaderData)  # note that this is a simpler function call than in C

    # This shader compilation is more explicit than the one used in
    # framework.cpp, which relies on a glutil wrapper function.
    # This is made explicit here mainly to decrease dependence on pyOpenGL
    # utilities and wrappers, which docs caution may change in future versions.
    glCompileShader(shader)

    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status == GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        strInfoLog = glGetShaderInfoLog(shader)
        strShaderType = ""
        if shaderType is GL_VERTEX_SHADER:
            strShaderType = "vertex"
        elif shaderType is GL_GEOMETRY_SHADER:
            strShaderType = "geometry"
        elif shaderType is GL_FRAGMENT_SHADER:
            strShaderType = "fragment"

        print("Compilation failure for " + strShaderType + " shader:\n" + str(strInfoLog))

    return shader


# Function that accepts a list of shaders, compiles them, and returns a handle to the compiled program
def createProgram(shaderList):
    program = glCreateProgram()

    for shader in shaderList:
        glAttachShader(program, shader)

    glLinkProgram(program)

    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status == GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        strInfoLog = glGetProgramInfoLog(program)
        print("Linker failure: \n" + str(strInfoLog))

    for shader in shaderList:
        glDetachShader(program, shader)

    return program


# Helper function to locate and open the target file (passed in as a string).
# Returns the full path to the file as a string.


def findFileOrThrow(strBasename):
    # Check if the file exists as is (absolute or relative path)
    if os.path.isfile(strBasename):
        return strBasename

    # Define the shader directory relative to this script's location
    SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    
    # Ensure strBasename only contains the file name
    strBasename = os.path.basename(strBasename)
    
    # Construct the full path
    strFilename = os.path.join(SHADER_DIR, strBasename)
    if os.path.isfile(strFilename):
        return strFilename

    # Raise an error if the file is not found
    raise IOError(f"Could not find target file {strBasename}. Looked in: {strFilename}")


