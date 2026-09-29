import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import { getModel, normalizeContext } from "../src/compat.ts";
import type { Model } from "../src/types.ts";

// Regression: a provider that answers 2xx and carries the failure in the response body.
// OpenRouter's free endpoints do this ("Provider returned an empty response",
// error_type: provider_unavailable), and the OpenAI SDK raises it while the stream is read —
// after `create()` has already resolved. Retrying only the request left that failure outside
// every retry policy, so a single empty provider response ended the turn with no retry.

const mockState = vi.hoisted(() => ({ createCalls: 0 }));

/** Mirrors the OpenAI SDK: APIError raised from the stream reader with no HTTP status. */
function bodyLevelProviderError(): Error & { status?: number; headers?: Headers } {
	const error = new Error("Provider returned an empty response") as Error & {
		status?: number;
		headers?: Headers;
	};
	error.status = undefined;
	error.headers = new Headers();
	return error;
}

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: () => {
					mockState.createCalls++;
					const attempt = mockState.createCalls;
					const stream = {
						async *[Symbol.asyncIterator]() {
							if (attempt === 1) throw bodyLevelProviderError();
							yield { choices: [{ delta: { content: "pong" } }] };
							yield { choices: [{ delta: {}, finish_reason: "stop" }] };
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}

	return { default: FakeOpenAI };
});

function createModel(): Model<"openai-completions"> {
	return getModel("openai", "gpt-4o-mini") as Model<"openai-completions">;
}

async function collect(maxRetries: number): Promise<{ text: string; calls: number }> {
	const context = normalizeContext({
		systemPrompt: "sys",
		messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
	});
	const response = streamOpenAICompletions(createModel(), context, {
		apiKey: "test-key",
		maxRetries,
	});
	let text = "";
	for await (const event of response) {
		if (event.type === "text_delta") text += event.delta;
	}
	await response.result();
	return { text, calls: mockState.createCalls };
}

describe("openai-completions body-level provider error", () => {
	beforeEach(() => {
		mockState.createCalls = 0;
	});

	it("retries a 200 response that fails in the body and completes the turn", async () => {
		const { text, calls } = await collect(1);

		expect(calls).toBe(2);
		expect(text).toBe("pong");
	});

	it("does not replay the request when the policy has no retries left", async () => {
		// The body-level failure now surfaces inside the retried region, so the only thing the
		// policy can do with no retries left is stop. What must never happen is a silent replay.
		const { text, calls } = await collect(0);

		expect(calls).toBe(1);
		expect(text).toBe("");
	});
});
