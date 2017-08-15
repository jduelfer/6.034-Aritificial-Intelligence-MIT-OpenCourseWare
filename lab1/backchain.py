from production import AND, OR, NOT, PASS, FAIL, IF, THEN, \
     match, populate, simplify, variables, RuleExpression
from zookeeper import ZOOKEEPER_RULES

# This function, which you need to write, takes in a hypothesis
# that can be determined using a set of rules, and outputs a goal
# tree of which statements it would need to test to prove that
# hypothesis. Refer to the problem set (section 2) for more
# detailed specifications and examples.

# Note that this function is supposed to be a general
# backchainer.  You should not hard-code anything that is
# specific to a particular rule set.  The backchainer will be
# tested on things other than ZOOKEEPER_RULES.


def backchain_to_goal_tree(rules, hypothesis):
	goal_tree = [hypothesis]
	for rule in rules:
		for consequent in rule.consequent(): # Extend from python list
			bound_variables = match(consequent, hypothesis)
			if bound_variables is not None: # We have matched expressions
				antecedent_list = rule.antecedent()
				if isinstance(antecedent_list, RuleExpression): # Complex expression
					if isinstance(antecedent_list, AND):
						inner_goal_tree = AND([backchain_to_goal_tree(rules, populate(expr, bound_variables))
												for expr in antecedent_list])
					elif isinstance(antecedent_list, OR):
						inner_goal_tree = OR([backchain_to_goal_tree(rules, populate(expr, bound_variables))
												for expr in antecedent_list])
					goal_tree.append(inner_goal_tree)
				else: # just a string, which is a single leaf node
					new_hypothesis = populate(antecedent_list, bound_variables)
					goal_tree.append(OR(backchain_to_goal_tree(rules, new_hypothesis)))

	return simplify(OR(goal_tree))


# Here's an example of running the backward chainer - uncomment
# it to see it work:
#print backchain_to_goal_tree(ZOOKEEPER_RULES, 'opus is a penguin')
