import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const originalQueuePrompt = api.queuePrompt;

async function checkIfElseBeforeQueuePrompt(number, originalPrompt) {
	app.graph.findNodesByType("ArcheryIfElse").forEach(node => {
        // Get the condition value
        let condition = node.widgets[0].value
        if (node.inputs.length === 3) {
            const conditionNode = getImmediateLinkedNode(node, 2)
            condition = conditionNode.widgets[0].value
        }

        // Mute / Unmute nodes based on condition
        const trueNode = getImmediateLinkedNode(node, 0)
        const falseNode = getImmediateLinkedNode(node, 1)
        if (trueNode)
            trueNode.mode = condition ? 0 : 2;
        if (falseNode)
            falseNode.mode = condition ? 2 : 0;
    })
    app.graph.change()
    const prompt = await app.graphToPrompt() ?? originalPrompt
	return originalQueuePrompt.call(api, number, prompt);
}

api.queuePrompt = checkIfElseBeforeQueuePrompt;

function getImmediateLinkedNode(node, inputIndex) {
    const nodeLinkId = node.inputs[inputIndex].link
    if (!nodeLinkId) return null        
    const nodeId = app.graph.links[nodeLinkId].origin_id;
    return app.graph.getNodeById(nodeId)
}
