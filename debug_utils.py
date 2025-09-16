import time

import onnxruntime
import torch
from torch.export import Dim

from e2e_implem.e2e_utils import format_detailed_error, save_error_report


def debug_dynamic_shapes_pipeline(model, example_input, model_name="model"):
    """
    Debug dynamic shapes through the entire pipeline:
    1. torch.export (most strict about dynamic shapes)
    2. torch.compile (intermediate validation)
    3. ONNX export (final conversion)
    """
    print(f"\n{'=' * 70}")
    print(f"DEBUGGING DYNAMIC SHAPES PIPELINE FOR: {model_name}")
    print(f"{'=' * 70}")

    # Define dynamic dimensions
    # batch_dim = Dim("batch", min=1, max=1)
    height_dim = Dim("height", min=32, max=2048)
    width_dim = Dim("width", min=32, max=2048)
    test_sizes = [(224, 224), (480, 640), (800, 600)]

    results = {}

    # STEP 1: Test torch.export first (most revealing)
    print("\n1️⃣ TESTING torch.export (MOST STRICT)")
    print("-" * 50)

    export_dynamic_shapes = None
    try:
        if isinstance(example_input, dict):
            export_dynamic_shapes = {
                "images": {2: height_dim, 3: width_dim},
                "post_process_kwargs": {"target_sizes": None},
            }
        elif isinstance(example_input, tuple):
            export_dynamic_shapes = (
                {"images": {2: height_dim, 3: width_dim}},
                {"post_process_kwargs": {"target_sizes": None}},
            )

        if not export_dynamic_shapes:
            raise ValueError("Unsupported example_input type for dynamic shape definition.")

        print("   🔄 Attempting torch.export with dynamic_shapes...")
        print(f"   📋 Dynamic shapes: {export_dynamic_shapes}")

        start_time = time.time()

        # Correctly pass args and kwargs to torch.export
        args = ()
        kwargs = {}
        if isinstance(example_input, dict):
            kwargs = example_input
        elif isinstance(example_input, tuple):
            args = example_input

        exported_program = torch.export.export(
            model,
            args,
            kwargs,
            dynamic_shapes=export_dynamic_shapes,
            strict=False,  # Allow some flexibility
        )
        export_time = time.time() - start_time

        print(f"   ✅ torch.export SUCCESS in {export_time:.2f}s")
        print(f"   📊 Exported graph has {len(list(exported_program.graph.nodes))} nodes")

        # Test with different input sizes using the exported program
        print("   🧪 Testing exported program with different sizes...")
        size_test_results = []

        for test_size in test_sizes:
            try:
                if isinstance(example_input, dict):
                    test_input_kwargs = {
                        "images": torch.randn(1, 3, test_size[0], test_size[1]),
                        "post_process_kwargs": {"target_sizes": torch.tensor([test_size[::-1]])},
                    }
                    _ = exported_program.module()(**test_input_kwargs)
                elif isinstance(example_input, tuple):
                    test_input_tuple = (
                        {
                            "images": torch.randn(1, 3, test_size[0], test_size[1]),
                            "post_process_kwargs": {"target_sizes": torch.tensor([test_size[::-1]])},
                        },
                    )
                    _ = exported_program.module()(*test_input_tuple)

                size_test_results.append(f"✅ {test_size}")
            except Exception as e:
                size_test_results.append(f"❌ {test_size}: {str(e)[:100]}...")
                if len(size_test_results) == 1:  # Only show detailed error for first failure
                    print(f"      📋 Detailed error for size {test_size}:")
                    format_detailed_error(e, f"- Size test {test_size}")

        print(f"   📈 Size test results: {', '.join(size_test_results)}")
        results["torch_export"] = "SUCCESS"

    except Exception as e:
        print(f"   ❌ torch.export FAILED: {str(e)}")
        results["torch_export"] = f"FAILED: {str(e)}..."

        # Get detailed error information
        error_details = format_detailed_error(e, "- torch.export")
        save_error_report(error_details, model_name, "torch_export")

        # Try to get more specific error info
        if "constraint violation" in str(e).lower():
            print("   💡 HINT: Constraint violation suggests model uses operations incompatible with dynamic shapes")
        elif "data dependent" in str(e).lower():
            print("   💡 HINT: Data-dependent operations detected - model logic depends on tensor values")
        elif "dynamic shape" in str(e).lower():
            print("   💡 HINT: Dynamic shape issue in model architecture")

    # STEP 2: Test torch.compile
    # print("\n2️⃣ TESTING torch.compile (INTERMEDIATE)")
    # print("-" * 50)

    # try:
    #     print("   🔄 Attempting torch.compile...")

    #     start_time = time.time()
    #     compiled_model = torch.compile(model, mode="default", dynamic=True)
    #     compile_time = time.time() - start_time

    #     print(f"   ✅ torch.compile SUCCESS in {compile_time:.2f}s")

    #     # Test compiled model with different sizes
    #     print("   🧪 Testing compiled model with different sizes...")
    #     compiled_size_results = []

    #     for test_size in test_sizes:
    #         try:
    #             if isinstance(example_input, dict):
    #                 test_input_kwargs = {
    #                     "images": torch.randn(1, 3, test_size[0], test_size[1]),
    #                     "post_process_kwargs": {"target_sizes": torch.tensor([test_size[::-1]])},
    #                 }
    #                 with torch.no_grad():
    #                     _ = compiled_model(**test_input_kwargs)
    #             elif isinstance(example_input, tuple):
    #                 test_input_tuple = (
    #                     {
    #                         "images": torch.randn(1, 3, test_size[0], test_size[1]),
    #                         "post_process_kwargs": {"target_sizes": torch.tensor([test_size[::-1]])},
    #                     },
    #                 )
    #                 with torch.no_grad():
    #                     _ = compiled_model(*test_input_tuple)

    #             compiled_size_results.append(f"✅ {test_size}")
    #         except Exception as e:
    #             compiled_size_results.append(f"❌ {test_size}: {str(e)[:100]}...")
    #             if len(compiled_size_results) == 1:  # Only show detailed error for first failure
    #                 print(f"      📋 Detailed error for size {test_size}:")
    #                 format_detailed_error(e, f"- Compiled size test {test_size}")

    #     print(f"   📈 Compiled size test results: {', '.join(compiled_size_results)}")
    #     results["torch_compile"] = "SUCCESS"

    # except Exception as e:
    #     print(f"   ❌ torch.compile FAILED: {str(e)}")
    #     results["torch_compile"] = f"FAILED: {str(e)[:50]}..."

    #     # Get detailed error information
    #     error_details = format_detailed_error(e, "- torch.compile")
    #     save_error_report(error_details, model_name, "torch_compile")

    # STEP 3: Test ONNX export (if previous steps succeeded)
    print("\n3️⃣ TESTING ONNX EXPORT (FINAL CONVERSION)")
    print("-" * 50)

    if results.get("torch_export") == "SUCCESS" or results.get("torch_compile") == "SUCCESS":
        print("   🔄 Attempting ONNX export...")

        try:
            print("     🔄 Trying Dynamo with dynamic_shapes...")
            safe_model_name = model_name.replace("/", "_")
            onnx_path = f"{safe_model_name}_{'dynamo_with_dynamic_shapes'.lower().replace(' ', '_')}.onnx"
            args = ()
            kwargs = {}
            input_names = []
            if isinstance(example_input, dict):
                kwargs = example_input
                input_names = list(example_input.keys())
            elif isinstance(example_input, tuple):
                args = example_input
                input_names = [f"input_{i}" for i in range(len(args))]

            onnx_program = torch.onnx.export(
                model,
                args=args,
                kwargs=kwargs,
                f=onnx_path,
                input_names=input_names,
                output_names=["output"],
                dynamo=True,
                dynamic_shapes=export_dynamic_shapes,
            )
            print("     ✅ Dynamo with dynamic_shapes SUCCESS")
            results["onnx_Dynamo with dynamic_shapes"] = "SUCCESS"

            # Optimize the ONNX model
            print("     🔄 Optimizing ONNX model...")
            onnx_program.optimize()
            print("     ✅ ONNX model optimized")
            onnx_program.save(onnx_path)
            print(f"     📋 ONNX model saved to {onnx_path}")

            # STEP 3.1: Verify ONNX model with onnxruntime
            print("     🧪 Verifying ONNX model with different sizes...")

            try:
                ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                onnx_input_names = [inp.name for inp in ort_session.get_inputs()]

                onnx_size_test_results = []

                for test_size in test_sizes:
                    try:
                        # Create test tensors
                        images_tensor = (torch.randn(1, 3, test_size[0], test_size[1]) * 255).to(torch.uint8)
                        target_sizes_tensor = torch.tensor([test_size[::-1]])

                        # Prepare inputs for ONNX Runtime.
                        onnx_inputs = {
                            onnx_input_names[0]: images_tensor.numpy(),
                        }
                        if len(onnx_input_names) > 1:
                            onnx_inputs[onnx_input_names[1]] = target_sizes_tensor.numpy()

                        ort_session.run(None, onnx_inputs)
                        onnx_size_test_results.append(f"✅ {test_size}")
                    except Exception as e:
                        onnx_size_test_results.append(f"❌ {test_size}: {str(e)[:100]}...")
                        if len(onnx_size_test_results) == 1:  # Only detail first error
                            format_detailed_error(e, f"- ONNX Runtime test {test_size}")

                print(f"     📈 ONNX Runtime test results: {', '.join(onnx_size_test_results)}")
                if any("❌" in s for s in onnx_size_test_results):
                    results["onnx_Dynamo with dynamic_shapes_runtime"] = "FAILED"
                else:
                    results["onnx_Dynamo with dynamic_shapes_runtime"] = "SUCCESS"

            except Exception as e:
                print(f"     ❌ ONNX Runtime verification FAILED: {str(e)}")
                results["onnx_Dynamo with dynamic_shapes_runtime"] = "FAILED"
                format_detailed_error(e, "- ONNX Runtime verification")

        except Exception as e:
            print(f"     ❌ Dynamo with dynamic_shapes FAILED: {str(e)}...")
            results["onnx_Dynamo with dynamic_shapes"] = f"FAILED: {str(e)}..."

            # Get detailed error information
            error_details = format_detailed_error(e, "- ONNX Dynamo with dynamic_shapes")
            save_error_report(
                error_details, model_name, f"onnx_{'dynamo_with_dynamic_shapes'.lower().replace(' ', '_')}"
            )
    else:
        print("   ⏭️  Skipping ONNX export - torch.export/compile failed")
        results["onnx_export"] = "SKIPPED - prerequisite failures"

    # STEP 4: Analysis and recommendations
    print("\n4️⃣ ANALYSIS & RECOMMENDATIONS")
    print("-" * 50)

    print("\n📊 RESULTS SUMMARY:")
    for step, result in results.items():
        status = "✅" if "SUCCESS" in result else "❌" if "FAILED" in result else "⏭️"
        print(f"   {status} {step}: {result}")

    print("\n💡 DIAGNOSIS:")
    if results.get("torch_export") == "SUCCESS":
        print("   ✅ Model architecture supports dynamic shapes")
        if "onnx" in results and any("SUCCESS" in v for k, v in results.items() if "onnx" in k):
            print("   ✅ ONNX conversion works - you should be able to use dynamic shapes!")
        else:
            print("   ⚠️  Model supports dynamic shapes but ONNX conversion fails")
            print(
                "   💡 Try: different opset versions, disable constant folding, or use torch.export → onnx conversion"
            )

    elif results.get("torch_compile") == "SUCCESS":
        print("   ⚠️  Model works with torch.compile but fails torch.export")
        print("   💡 Issue likely: data-dependent operations or unsupported dynamic patterns")
        print("   💡 Try: simplify model, remove conditional logic, or use torch.compile for deployment")

    else:
        print("   ❌ Model architecture fundamentally incompatible with dynamic shapes")
        print("   💡 Solutions: model surgery, export backbone only, or use multiple fixed-size models")

    print(f"\n{'=' * 70}")
    return results
