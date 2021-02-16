"""
Anthony Correia
02/01/21
Some global variables
"""

from numpy import log


from .config import loc, default_fontsize

import importlib.util
spec = importlib.util.spec_from_file_location("definition_project", loc['definition'])
definition_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(definition_project)
definition_quantities = definition_project.definition_quantities
latex_particles = definition_project.latex_particles


import HEA.tools.assertion as assertion
from HEA.tools.da import el_to_list
from HEA.tools.string import add_text, _latex_format, string_between_brackets
from HEA.tools.serial import get_latex_column_table


# Name of the functions and functions ======================================================

# Name of the functions and functions
definition_functions = {
    'ln(1-x)' : (lambda x: log(1.-x)),
    'ln(x)'   : (lambda x: log(x)),
    'x/y'     : (lambda x: x[0]/x[1]),
    'identity': (lambda x: x),
    None      : None
}

latex_functions = {
    'x/y'     : (lambda x: f'$\\frac{{{x[0]}}}{{{x[1]}}}$')
}

# Particles =================================================================================


class RVariable():
    ## Constructors ------------------------------------------------------------
    def __init__(self, raw_quantity, particle, name_function=None):
        """ Create a RVariable from a physical quantity, a particle and the function applied to the variable.
        
        Parameters
        ----------
        raw_quantity: str or list(str) or tuple(str)
            A physical quantity (e.g., `'M'` the invariant mass, ...)  or list of physical quantities
            If it is an iterable, the `function` must transform them into a single variable (i.e., the function takes more than 1 variable)
        particle: str or list(str) or tuple(str)
            A particle (e.g., 'B0' for the B0 meson) or a list of particles.
            - If `particle` is a list, `quantity` must be a list of same length
            - If `quantity` is a list and `particle` is a str, then each physical quantity refers to the same `particle`
        function: str or None
            Name of the functions (as defined in the dictionnary `definition_functions`) that can be applied to the quantities
        
        """
        assertion._assert_is_in_iterable(name_function, definition_functions)
        
        if particle is not None and not isinstance(particle, str):
            assertion._assert_list_tuple(particle)
            assertion._assert_list_tuple(raw_quantity)
            
            particle = tuple(particle)
            raw_quantity = tuple(raw_quantity)
            
            assertion._assert_same_length(particle, raw_quantity)
            
        elif not isinstance(raw_quantity, str):
            assertion._assert_list_tuple(raw_quantity)
            raw_quantity = tuple(raw_quantity)
        
        if not isinstance(raw_quantity, str):
            # if raw_quantity is an iterable, the `function` must transform them into a single variable (i.e., the function takes more than 1 variable)
            assertion._assert_not_none(name_function)        
        
        self.raw_quantity = raw_quantity
        self.particle = particle        
        self.name_function = name_function
    
    @staticmethod
    def from_branch(branch):
        """ Create a RVariable from a branch
        Parameters
        ----------
        branch: branch name
        
        Returns
        -------
        RVariable:
            RVariable with branch given by the argument
        """
        particle, raw_quantity, name_function = RVariable.get_particle_raw_quantity_name_function_from_branch(branch)
        assert raw_quantity is not None
        return RVariable(particle=particle, raw_quantity=raw_quantity, name_function=name_function)
    
    
    def __str__(self):
        return self.latex_branch
    
    
    ## non-latex properties -----------------------------------------------------------
    
    
    @property
    def quantity(self):
        """ Combined physical quantity, with the function applied to it. Only defined if there is zero or one particle.
        - `{raw_quantity}` if there is one raw branch and no function
        - `{raw_quantity}:{name_function}` if there is one raw quantity and a function
        - `{raw_quantity[0]},{raw_quantity[1]}:{name_function}` if there are two raw variables and a function
        """
        if assertion.is_list_tuple(self.particle) and len(self.particle) > 1:
            return None
        else:
            if isinstance(self.raw_quantity, str):
                raw_quantities = [self.raw_quantity]
            else:
                raw_quantities = self.raw_quantity
            return add_text(','.join(raw_quantities), self.name_function, sep=':')
            
    
    @property
    def raw_branch(self):
        """ The name of the branch that can be found in a root file
        - `{raw_quantity}` if there is one raw quantity, zero particle
        - `raw_variable = {particle}_{raw_quantity}` if there is one raw quantity, one particle
        - Tuple of `tuple({raw_variable[0]},{raw_variable[1]})` if there is two raw variables
        - ...
        """
        return RVariable.get_raw_branch(self.particle, self.raw_quantity)
        
    
    @property
    def branch(self):
        """ The name of the branch used in the pandas dataframe
        - `{raw_branch}` if there is one raw branch and no function 
        - `{raw_branch}:{name_function}` if there is one raw quantity and a function 
        - `{raw_branch[0]},{raw_branch[1]}:{name_function}` if there is two raw variables and a function 
        """
        return  self.get_branch_from_raw_branch_name_function(self.raw_branch, self.name_function)
        
    
    @property
    def function(self):
        return definition_functions[self.name_function]
    
    @property
    def unit(self):
        """ unit of the physical quantity
        Check in the `definition_quantities` dictionnary with `self.quantity` or `self.branch` as a key
        If their is nothing, just return None.
        """
        if self.branch in definition_quantities:
            return definition_quantities[self.branch]['unit']
        elif self.quantity in definition_quantities:
            return definition_quantities[self.quantity]['unit']
        else:
            return None
    
    ## Latex properties -----------------------------------------------------------
    
    @property
    def latex_particle(self):
        return self.get_latex_particle(self.particle)
    
    @property
    def latex_raw_quantity(self):
        return self.get_latex_raw_quantity(self.raw_quantity)
    
    @property
    def latex_quantity(self):
        """
        1. If there are more than 1 particle, return None. The latex name of the quantity is not well defined in this case!
            just return None
        2. if there is no function, just return the `latex_quantity_without_function` argument
        3. if there is a function:
            3.1. The latex name might be provided in the `definition_quantities` dictionnary with the key `self.quantity` (if not None)
            3.2. If there is a function, if the dictionnary `latex_functions` contains the name of the function, it means that another way to specify the application of the function to the variable is provided, it has to be used in preference. For instance, we have chosen to show the division function with the `\frac{}{}` latex operator.
            3.3. If the latex name is not provided in this dictionnary, the latex name is just `"{name_function} of the {latex_raw_quantity}"` or `"{name_function} of the {latex_raw_quantity[0]} and the {latex_raw_quantity[1]}"` if there are more than 1 quantity
        """
        # Case 1
        if assertion.is_list_tuple(self.particle) and len(self.particle) > 1:
            return None
        
        # Case 2
        elif self.name_function is None:
            return self.latex_raw_quantity
        
        # Case 3
        else:
            return self.__get_latex_variable(self.quantity, self.latex_raw_quantity, self.name_function)
#             # Case 3.1
#             if self.quantity in definition_quantities:
#                 return definition_quantities[self.quantity]['latex']
            
#             # Case 3.2:
#             elif self.name_function in latex_functions:
#                 return latex_functions[self.name_function](self.latex_raw_quantity)
            
#             # Case 3.3
#             else:                    
#                 latex_raw_quantities_text = ' and the '.join(self.latex_raw_quantity)
#                 return f"{self.name_function} of the {latex_raw_quantities_text}"
                
    
    @property
    def latex_raw_branch(self):
        """ Return the latex name of the branches, without including the function in the label
        Returns
        -------
        latex_raw_branch: str or tuple(str)
            - if `self.raw_branch` is one branch, returns `"{self.latex_raw_quantity}({self.latex_particle})"` or just `self.latex_quantity` if no the rVariable has no particle.
            - if `self.latex_quantity` is a tuple of quantities, returns a tuple of latex names of variables.
        """
        return self.get_latex_raw_branch(self.latex_particle, self.latex_raw_quantity)
    
    @property
    def latex_branch(self):
        """ Full latex name of the variable
        1. If there is no function specified, the latex name of the variable is merely `self.latex_raw_branch` 
        2. The latex name might be provided in the `definition_quantities` dictionnary with the key `self.branch`
        3. If there is one particle for several quantities, specify the name of the particle just at the very end
            --> return `{latex_quantity}({latex_particle})`
        4. If there is the same number of particles and quantities, specify the particle for each variable
            1. If there is a function, if the dictionnary `latex_functions` contains the name of the function, it means that another way to specify the application of the function to the variable is provided, it has to be used in preference. For instance, we have chosen to show the division function with the `\frac{}{}` latex operator.
            2. If the dictionnary does not contain a label for the function: if there is only one variable the name of the variable is `"{name_function} of the {latex_raw_branch}"` or `"{name_function} of the {latex_raw_branch[0]} and {latex_raw_branch[1]}"` if there are more than one variable.
        
        
        
        These steps are done with ` self.__get_latex_variable(variable, latex_raw_variable, name_function)`
        """
        # Case 1
        if self.name_function is None:
            return self.set_latex_upper_case(self.latex_raw_branch)
        
        # Case 2
        if self.branch in definition_quantities:
            return definition_quantities[self.branch]['latex']
        
        
        are_one_particle_and_several_quantities = assertion.is_list_tuple(self.raw_quantity) and len(self.raw_quantity)>1 and isinstance(self.particle, str)
        # Case 3
        if are_one_particle_and_several_quantities:
            return f"{self.latex_quantity}({self.latex_particle})"
        
        #Case 4
        else:
            return self.__get_latex_variable(None, self.latex_raw_branch, self.name_function)
    ## Static methods ---------------------------------------------------------------
    
    @staticmethod
    def get_raw_branch(particle, raw_variable):
        """
        Parameters
        ----------
         particle: str or tuple(str) or None
            particle or list of particles
        raw_variable: str
            name of the raw variable
        Returns
        -------
        raw_branch: str or tuple(str)
            Name of the raw branch or tuple of the names of the raw branches
            - `{raw_quantity}` if there is one raw quantity, zero particle
            - `raw_variable = {particle}_{raw_quantity}` if there is one raw quantity, one particle
            - Tuple of `tuple({raw_variable[0]},{raw_variable[1]})` if there is two raw variables
            - ...
        """
        
        if assertion.is_list_tuple(raw_variable):
            particle = tuple(el_to_list(particle, len(raw_variable)))
            return tuple(RVariable.get_raw_branch(sub_particle, sub_raw_quantity) for sub_particle, sub_raw_quantity in zip(particle, raw_variable))
        
        return add_text(particle, raw_variable) 
    
    @staticmethod
    def get_latex_particle(particle):
        """ Get the latex label of a particle from its raw name
        Parameters
        ----------
        particle: str or tuple(str) or None
            particle or list of particles
            
        
        Returns
        -------
        latex_particle: str or tuple(str) or None
            - latex name of the particle(s) if specified in `latex_particles`
            - else, `particle`
            - if a list of particle is provided, returns a tuple
        """
        
        # If `particle` is a list (recursion)
        if assertion.is_list_tuple(particle):
            return tuple(RVariable.get_latex_particle(sub_particle) for sub_particle in particle)
        
        if particle is None:
            return None
        elif particle in latex_particles:
            return latex_particles[particle]
        else:
            return particle
    
    @staticmethod
    def get_latex_raw_quantity(raw_quantity):
        """ Get the latex label of a raw quantity from its name
        Parameters
        ----------
        raw_quantity: str or tuple(str)
            raw quantity or list of raw quantities
            
        
        Returns
        -------
        latex_raw_quantity: str or tuple(str) or None
            - latex name of the raw quantitie(s) if specified in `definition_quantities`
            - else, `raw_quantity`
            - if a list of raw quantities is provided, returns a tuple of their latex names
        """
        assertion._assert_not_none(raw_quantity)
        
        # If `raw_quantity` is a listraw_quantity(recursion)
        if assertion.is_list_tuple(raw_quantity):
            return tuple(RVariable.get_latex_raw_quantity(sub_raw_quantity) for sub_raw_quantity in raw_quantity)
        
        
        elif raw_quantity in definition_quantities:
            return definition_quantities[raw_quantity]['latex']
        else:
            return _latex_format(raw_quantity)
    
    @staticmethod
    def get_latex_raw_branch(latex_particle, latex_raw_quantity):
        """ Return the latex name of the raw branches (i.e., quantity and particle), without including the function in the label
        Parameters
        ----------
        latex_particle: str or list(str)
            latex name of particle or list of latex names of particles
        latex_raw_quantity: str or list(str)
            latex name of a raw quantity or list of latex names of raw quantities
        Returns
        -------
        latex_raw_branch: str or tuple(str)
            - if `self.raw_branch` is one branch, returns `"{self.latex_quantity}({self.latex_particle})"` or just `self.latex_quantity` if no the rVariable has no particle.
            - if `self.latex_quantity` is a tuple of quantities, returns a tuple of latex names of variables.
        """
        
        # If `latex_raw_quantity` is a list (recursion)
        if assertion.is_list_tuple(latex_raw_quantity):
            # Repeat the name of the particle in the case it is the same (i.e., there is one particle specified but several quantities)
            latex_particles = tuple(el_to_list(latex_particle, len(latex_raw_quantity)))
            return tuple(RVariable.get_latex_raw_branch(sub_latex_particle, sub_latex_raw_quantity) for sub_latex_particle, sub_latex_raw_quantity in zip(latex_particles, latex_raw_quantity))
        
        
        if latex_particle is None:
            return latex_raw_quantity
        else:
            return f"{latex_raw_quantity}({latex_particle})"
    
    
    @staticmethod
    def get_particle_raw_quantity_from_raw_branch(raw_branch):
        """ Get the name of the particle and of the raw physical quantity from the raw branch name
        
        Parameters
        ----------
        branch : str
            name of the branch (for instance: 'B0_M')
            (cannot be a tuple in this function!)

        Returns
        -------
        particle : str
            name of the particle ('B0', 'tau', ...), key in the dictionnary latex_particles
        raw_quantity : str
            name of the raw physical quantity ('P', 'M', ...)

        Hypothesis
        ----------
        the particle is in the dictionnary latex_particles in `{loc['project']}/definition.py`
        
        Examples
        --------
        >> get_particle_raw_quantity_from_branch('B0_M')
        ('B0', 'M')
        >> get_particle_raw_quantity_from_branch('BDT')
        (None, 'BDT')
        """
        
        if assertion.is_list_tuple(raw_branch):
            return tuple(RVariable.get_particle_raw_quantity_from_branch(sub_raw_branch) for sub_raw_branch in raw_branch)
        
        list_particles = list(latex_particles.keys())
    
        ## Get the name of the particle and deduce the name of the var
        particle = raw_branch
        marker = None
        marker_before = 0

        # We must have: branch = `"{particle}_{raw_quantity}"`
        # As long as particle is not in the list of the name of particles
        # get the next '_' (starting from the end)
        # cut branch in before '_' and after '_'
        # What is before should be the name of the particle
        # unless we need to select the next '_' to get the name of the particle
        # because this '_' is part of the name of the raw quantity
        while particle not in list_particles and marker != marker_before:
            marker_before = marker
            cut_branch = raw_branch[:marker]
            marker = len(cut_branch) - 1 - cut_branch[::-1].find('_') # find the last '_'
            particle = raw_branch[:marker]

        # if there were a '_' in branch, we assume the separation was done
        if marker != marker_before:
            raw_quantity = raw_branch[marker + 1:]
            return particle, raw_quantity

        # if we did not find a particle : the raw quantity is the raw branch itself
        else:
            return None, raw_branch
    
    @staticmethod
    def get_particle_raw_quantity_name_function_from_branch(branch):
        """ Get the particle, the raw quantity and the function from the name branch
        Parameters
        ----------
        branch : str
            branch name

        Returns
        -------
        particle     : str
            Particle name (e.g., 'B0', 'Dst', ...)
        raw_quantity : str or tuple(str)
            name of the raw quantity or tuple of raw quantities (e.g., 'M', 'BDT', ...)
        name_function : str
            name of the function applied to the data
        """
        
        # function
        if ':' in branch:
            full_raw_branch, name_function = branch.split(':')
        else:
            name_function = None
            full_raw_branch = branch
        
        # full_raw_branch might be 'B0_M', 'BDT', or if there are more than 1 variable, 'B0_M,BDT' 
        # We get all the variables separately
        raw_branches = full_raw_branch.split(',')
        
        # We find the particle and raw_quantity of each
        # We check if they all refer to the same particle
        particles = [None]*len(raw_branches)
        raw_quantities = [None]*len(raw_branches)
        
        common_particle = True
        
        for i, raw_branch in enumerate(raw_branches):
            particles[i], raw_quantities[i] = RVariable.get_particle_raw_quantity_from_raw_branch(raw_branch)
            
            common_particle = common_particle and (particles[i]==particles[0])
        
        particles = particles[0] if common_particle else tuple(particles)
        if len(raw_quantities)==1:
            raw_quantities = raw_quantities[0] 
        return particles, raw_quantities, name_function
        
    @staticmethod
    def get_latex_raw_variable_unit_from_particle_raw_quantity(particle, raw_quantity):
        """ Get the latex label of the particle, the variable and its unit, from the couple (particle, raw_quantity)
        Parameters
        ----------
        particle     : str or None
            name of the particle
        raw_quantity : str or None
            name of the raw quantity

        Returns
        -------
        latex_particle     : str
            Latexl label of the particle (is returned only if get_particle is True)
        latex_raw_variable : str
            Latex label of the raw variable
        unit : str
            unit of the raw quantity
        """
        if particle is not None and particle in latex_particles:
            name_particle = latex_particles[particle]
        else:
            name_particle = None

        if raw_quantity is not None:

            if variable in variables_params:
                latex_raw_variable = definition_variables[variable]['name']
                unit = definition_variables[variable]['unit']
            else:
                latex_raw_variable = None
                unit = None

            if name_var is None:
                latex_raw_variable = _latex_format(variable)
            if name_particle is None:
                latex_raw_variable = _latex_format(variable)
            else:
                if get_particle:
                    latex_raw_variable = name_variable
                else:
                    latex_raw_variable = f"{name_variable}({name_particle})"

        else: # variable is None
            name_variable = branch
            unit = None

        return latex_raw_variable, unit
    
    @staticmethod
    def __get_latex_variable(variable, latex_raw_variable, name_function):
        """ 
        3.1. The latex name might be provided in the `definition_quantities` dictionnary with the key `raw_variable` (if not None)
        3.2. If there is a function, if the dictionnary `latex_functions` contains the name of the function, it means that another way to specify the application of the function to the variable is provided, it has to be used in preference. For instance, we have chosen to show the division function with the `\frac{}{}` latex operator.
        3.3. If the latex name is not provided in this dictionnary, the latex name is just `"{name_function} of the {latex_raw_variable}"` or `"{name_function} of the {latex_raw_variable[0]} and {latex_raw_variable[1]}"` if there are more than 1 quantity
        Parameters
        ----------
        variable: str
            name of the variable
        latex_raw_variable: str
            latex name(s) of the raw variables (no function applied to it)
        name_function: str
            name of the function applied to the variable
        
        Returns
        -------
        latex_variable: str
            latex name of the varaible
        """
        # Case 3.1
        if variable in definition_quantities:
            return definition_quantities[variable]['latex']
        
        # Case 3.2:
        elif name_function in latex_functions:
            return latex_functions[name_function](latex_raw_variable)
        
        # Case 3.3
        else:
            latex_raw_variables = el_to_list(latex_raw_variable, 1)
            latex_raw_variables_text = ' and '.join(latex_raw_variables)
            return f"{name_function} of the {latex_raw_variables_text}"
    
    
    @staticmethod
    def set_latex_upper_case(string):
        """
        Parameters
        ----------
        string: str
        
        Returns
        -------
        str
            string with a upper case for the first letter if
                - the first character is not `$` (latex)
                - the second character is not an upper case (to take into account names such as `sPlot`
        """
        if not string.startswith('$') and string[-1].isupper():
            string = string[0].upper() + string[1:]
        return string
    
    ## STATIC METHODS FOR USE OUTSIDE =========================================================
    
    @staticmethod
    def get_latex_branch_unit_from_branch(branch):
        """ Get the latex name and the unit associated with a branch.
        
        Parameters
        ----------
        branch : str
        
        Returns
        -------
        latex_branch: str
            latex name of the branch
        unit : str
            unit of the branch
        """
        rVariable = RVariable.from_branch(branch)
        return rVariable.latex_branch, rVariable.unit
        
    
         
    
    @staticmethod
    def get_branch_from_raw_branch_name_function(raw_branch, name_function=None):
        """ Get the raw name of the branch used in this library

        Parameters
        ----------
        raw_branch    : str or tuple(str)
            name of a branch or tuple of name of branches 
        name_function : str or None
            name of the function

        Examples
        --------
        > _get_name_var('branch', 'function')
        'branch:function'
        > _get_name_var(['branch1', 'branch2'], 'function')
        'branch1,branch2:function'
        > _get_name_var('branch')
        'branch'
        """
        if name_function is None:
            assert isinstance(raw_branch, str)
            return raw_branch
        else:
            raw_branches = el_to_list(raw_branch, 1)
            return ",".join(raw_branches) + ":" + name_function
    

def print_used_branches_per_particles(branches):
    """ Print a latex table with the used branches per particle.
    If a particle involves exactly the sames branches, gather the particles together
    
    Parameters
    ----------
    branches : list(str)
        list of used branches
    """
    branches_per_particles = {}

    # Retrieve the variables in the form of a dictionnary particle : list of variables
    for sub_branch in branches:
        rVariable = RVariable.from_branch(sub_branch)
        particle = rVariable.particle
        branch   = rVariable.branch

        if branch == '$p_T$':
            branch = 'Transverse momentum ' + branch

        if particle not in branches_per_particles:
            branches_per_particles[particle] = set()

        branches_per_particles[particle] = branches_per_particles[particle] | {branch}


    # gather the particles that have the same set of variables
    branches_per_particles_short = {}

    for particle, particle_branches in branches_per_particles.items():
        saved = False

        for saved_particle, saved_particle_branches in branches_per_particles_short.items():
            # check if the set of branches is already saved somewhere
            if particle_branches == saved_particle_branches:
                if not isinstance(saved_particle, tuple):
                    saved_particle_tuple = tuple([saved_particle])
                else:
                    particle_saved_tuple = particle_saved
                branches_per_particles_short[particle_saved_tuple + tuple([particle])] = particle_branches
                del branches_per_particles_short[saved_particle]
                saved = True
                break

        if not saved:
            branches_per_particles_short[particle] = particle_branches

    # Built the table
    print('\\begin{tabular}{ll}')
    print('Particle & Variable \\\\ \\hline \\hline')
    for particle, particle_branches in branches_per_particles_short.items():
        row_latex = get_latex_column_table(particle)
        row_latex+= ' & '
        row_latex+= get_latex_column_table(particle_branches)
        row_latex+= ' \\\\ \hline'
        print(row_latex)
    print('\\end{tabular}')
    
def get_branches_from_raw_branches_functions(raw_branches_functions):
    """ Turn a list of raw branches with the functions that will be applied to each quantity afterwards,
    into the list of the branches

    Parameters
    ----------
    raw_quantities_functions :  list(str or (str, str) or (tuple, str)
        list of 
            - branches
            - tuple (branch, name_function), where name_function is the name of the function applied to the branch
            - tuple (raw branches, name_function), where branches is a tuple of the raw branches and name_function is the name of the function applied to the raw branches.

    Returns
    -------
    branches: list 
        list of all the branches
    """
    branches = []
    for raw_branch_function in raw_branches_functions:
        if isinstance(raw_branch_function, tuple):
            raw_branch = raw_branch_function[0]
            name_function = raw_branch_function[1]
        else:
            raw_branch = raw_branch_function
            name_function = None

        branch = RVariable.get_branch_from_raw_branch_name_function(raw_branch, name_function)
        branches.append(branch)

    return branches
    
    
    
def get_raw_branches_from_raw_branches_functions(raw_branches_functions):
    """ turn a list of raw branches with the functions that will be applied to each quantity afterwards,
    into the list of the needed branches that need to be loaded from a root file.

    Parameters
    ----------    
    raw_quantities_functions :  list(str or (str, str) or (tuple, str)
            list of 
                - branches
                - tuple (branch, name_function), where name_function is the name of the function applied to the branch
                - tuple (raw branches, name_function), where branches is a tuple of the raw branches and name_function is the name of the function applied to the raw branches.

    returns
    -------
    raw_branches: list
        list of all the needed branches that need to be loaded from the root files
        in order to create the branches specified in `branches_functions`
    """
    raw_branches = []
    for raw_branch_function in raw_branches_functions:
        if isinstance(raw_branch_function, tuple):
            raw_branch = raw_branch_function[0]
        else:
            raw_branch = raw_branch_function

        if isinstance(raw_branch, tuple):
            raw_branches += raw_branch
        else:
            raw_branches.append(raw_branch)

    return raw_branches      